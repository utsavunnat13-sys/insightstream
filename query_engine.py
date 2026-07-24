import os
import json
import re
import pandas as pd
import numpy as np
import google.generativeai as genai
from openai import OpenAI
import anthropic
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
import dotenv
from settings import load_settings

# Load environment variables from .env file
dotenv.load_dotenv()

# LLM clients configured dynamically per request

class QueryResponseSchema(BaseModel):
    explanation: str = Field(description="Explanation of the data analysis approach and result.")
    python_code: str = Field(description="Executable Python code that operates on a DataFrame 'df' and assigns the final result to 'result_df'.")
    chart_type: str = Field(description="Type of chart to display: 'bar', 'line', 'pie', 'scatter', or 'none'.")
    x_key: Optional[str] = Field(description="Column name for the X-axis of the chart (required if chart_type is not 'none').")
    y_keys: Optional[List[str]] = Field(description="List of column names for the Y-axis / values (required if chart_type is not 'none').")

def clean_code(code: str) -> str:
    """Removes markdown code blocks if present."""
    code = code.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()

def execute_pandas_code(code_str: str, df: pd.DataFrame) -> pd.DataFrame:
    """Safely executes the generated code within a controlled local context."""
    code_str = clean_code(code_str)
    
    # Set up execution environment
    local_scope = {
        "pd": pd,
        "np": np,
        "df": df.copy()
    }
    
    # Execute the code
    try:
        exec(code_str, {}, local_scope)
    except Exception as e:
        raise RuntimeError(f"Failed to execute code: {e}\nGenerated Code:\n{code_str}")
        
    if "result_df" not in local_scope:
        raise ValueError("The generated Python code did not define 'result_df'.")
        
    res = local_scope["result_df"]
    
    # Ensure it's a DataFrame
    if isinstance(res, pd.Series):
        res = res.to_frame().reset_index()
    elif not isinstance(res, pd.DataFrame):
        try:
            res = pd.DataFrame(res)
        except Exception:
            res = pd.DataFrame([{"value": res}])
            
    # Clean NaN/Inf values so they are JSON-serializable
    res = res.replace([np.inf, -np.inf], np.nan)
    res = res.where(pd.notnull(res), None)
    
    return res

def run_insight_query(query: str, df: pd.DataFrame, mode: str = "structured", filename: Optional[str] = None) -> Dict[str, Any]:
    """Sends the schema and query to the selected LLM provider, executes the returned code, and returns the result."""
    
    if mode == "rag":
        return run_rag_query(query, df, filename)
    
    # 1. Prepare Schema metadata to feed to the LLM
    columns_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        # Sample values
        sample_vals = df[col].dropna().head(3).tolist()
        columns_info.append(f"- {col} ({dtype}). Sample: {sample_vals}")
        
    schema_str = "\n".join(columns_info)
    df_shape = df.shape
    
    prompt = f"""You are an expert Data Analyst and Python Developer.
We have a Pandas DataFrame named 'df' loaded from a CSV file.

### DataFrame Information:
- Shape: {df_shape[0]} rows, {df_shape[1]} cols
- Columns and types:
{schema_str}

### User's Question:
"{query}"

### Your Instructions:
1. Write a clean Python script using Pandas to answer the user's question.
2. The input dataframe is available as 'df'.
3. Save your final output in a variable named 'result_df'.
4. Do NOT import pandas or numpy (they are already imported as 'pd' and 'np').
5. Do NOT include markdown code blocks in the 'python_code' JSON field; write it as a raw string.
6. Determine the most appropriate chart to represent the final 'result_df' data:
   - 'bar': for categorical comparisons.
   - 'line': for trends over time/sequences.
   - 'pie': for parts of a whole (under 7 categories).
   - 'scatter': for numeric correlation comparisons.
   - 'none': if it's a single value, text summary, or not easily visualized.
7. Return a JSON object matching this schema:
{{
  "explanation": "Brief explanation of what the analysis shows.",
  "python_code": "result_df = ...",
  "chart_type": "bar|line|pie|scatter|none",
  "x_key": "column_for_x_axis",
  "y_keys": ["column_for_y_axis_1", "column_for_y_axis_2"]
}}
"""
    
    # Load current active settings
    app_settings = load_settings()
    provider = app_settings.provider
    
    # Check if we should use the local rule-based engine
    if provider == "local":
        res = get_fallback_mock_response(query, df)
        res["has_api_key"] = False
        res["provider"] = "local"
        return res

    try:
        parsed_response = None
        
        if provider == "gemini":
            api_key = app_settings.gemini_api_key
            if not api_key:
                raise ValueError("Gemini API Key not found. Please add it in Settings.")
            genai.configure(api_key=api_key)
            model_name = app_settings.gemini_model or "gemini-3.5-flash"
            model = genai.GenerativeModel(model_name)
            
            response = model.generate_content(
                prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": QueryResponseSchema,
                }
            )
            parsed_response = json.loads(response.text)
            
        elif provider == "openai":
            api_key = app_settings.openai_api_key
            if not api_key:
                raise ValueError("OpenAI API Key not found. Please add it in Settings.")
            client = OpenAI(api_key=api_key)
            model_name = app_settings.openai_model or "gpt-4o-mini"
            
            response = client.beta.chat.completions.parse(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                response_format=QueryResponseSchema,
            )
            parsed_model = response.choices[0].message.parsed
            if parsed_model:
                parsed_response = parsed_model.model_dump()
            else:
                raw_content = response.choices[0].message.content or ""
                parsed_response = json.loads(raw_content)
                
        elif provider == "anthropic":
            api_key = app_settings.anthropic_api_key
            if not api_key:
                raise ValueError("Anthropic API Key not found. Please add it in Settings.")
            client = anthropic.Anthropic(api_key=api_key)
            model_name = app_settings.anthropic_model or "claude-3-5-sonnet-latest"
            
            response = client.messages.create(
                model=model_name,
                max_tokens=4000,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                tools=[
                    {
                        "name": "respond_with_analysis",
                        "description": "Respond with the structured analysis output.",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "explanation": {"type": "string", "description": "Explanation of the data analysis approach and result."},
                                "python_code": {"type": "string", "description": "Executable Python code that operates on a DataFrame 'df' and assigns the final result to 'result_df'."},
                                "chart_type": {"type": "string", "enum": ["bar", "line", "pie", "scatter", "none"], "description": "Type of chart to display."},
                                "x_key": {"type": "string", "description": "Column name for the X-axis of the chart (null if none)."},
                                "y_keys": {"type": "array", "items": {"type": "string"}, "description": "List of column names for the Y-axis / values."}
                            },
                            "required": ["explanation", "python_code", "chart_type"]
                        }
                    }
                ],
                tool_choice={"type": "tool", "name": "respond_with_analysis"}
            )
            
            tool_use = next(block for block in response.content if block.type == "tool_use")
            parsed_response = tool_use.input
            
        elif provider == "ollama":
            api_base = app_settings.ollama_api_base or "http://localhost:11434/v1"
            model_name = app_settings.ollama_model or "llama3"
            
            client = OpenAI(base_url=api_base, api_key="ollama")
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            raw_content = response.choices[0].message.content or ""
            parsed_response = json.loads(raw_content)
            
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
            
        if not parsed_response:
            raise ValueError("No response received from LLM provider.")
            
        code_to_run = parsed_response.get("python_code") or parsed_response.get("pythonCode") or ""
        explanation = parsed_response.get("explanation") or ""
        chart_type = parsed_response.get("chart_type") or parsed_response.get("chartType") or "none"
        x_key = parsed_response.get("x_key") or parsed_response.get("xKey")
        y_keys = parsed_response.get("y_keys") or parsed_response.get("yKeys") or []
        
        # Execute the pandas code
        result_df = execute_pandas_code(code_to_run, df)
        
        # Convert result_df to list of dicts for JSON transmission
        chart_data = result_df.to_dict(orient="records")
        
        return {
            "explanation": explanation,
            "python_code": code_to_run,
            "chart_type": chart_type,
            "x_key": x_key,
            "y_keys": y_keys,
            "data": chart_data,
            "columns": list(result_df.columns),
            "success": True,
            "has_api_key": True,
            "provider": provider
        }
        
    except Exception as e:
        print(f"Error in query execution ({provider}): {e}")
        # Fall back to local engine on error, but report the warning
        fallback_res = get_fallback_mock_response(query, df)
        fallback_res["error"] = str(e)
        fallback_res["explanation"] = f"Warning: Provider '{provider}' failed ({e}). Fell back to Local Heuristic Engine.\n\n" + fallback_res["explanation"]
        fallback_res["has_api_key"] = False
        fallback_res["provider"] = "local"
        return fallback_res

# Lazy loaded SentenceTransformer and pre-computed template embeddings
_model = None
_template_embeddings = None

TEMPLATES = [
    {
        "intent": "grouped_aggregation",
        "description": "Calculate average, mean, sum, total, min, max of a numeric column grouped by a categorical column",
        "examples": [
            "average expected salary by department",
            "mean salary per role",
            "total interview score by status",
            "minimum experience for each role",
            "maximum salary by department",
            "sum of expected salary by department",
            "average expected salary grouped by department"
        ]
    },
    {
        "intent": "grouped_count",
        "description": "Count occurrences of items grouped by a categorical column",
        "examples": [
            "count candidates by status",
            "number of applicants per department",
            "how many candidates in each role",
            "distribution of status",
            "breakdown of department candidates",
            "count candidates grouped by role"
        ]
    },
    {
        "intent": "correlation",
        "description": "Compare two numeric columns using a scatter plot or correlation",
        "examples": [
            "years of experience vs expected salary",
            "interview score versus years of experience",
            "scatter plot of experience and salary",
            "correlation between experience and score",
            "experience vs expected salary"
        ]
    },
    {
        "intent": "summary_statistics",
        "description": "Get descriptive summary statistics of the dataset",
        "examples": [
            "show me a summary of the dataset",
            "describe the data",
            "statistical profile of the columns",
            "stats of the columns",
            "summarize the dataset"
        ]
    },
    {
        "intent": "preview_dataset",
        "description": "Show a preview or list the first few rows of the dataset",
        "examples": [
            "preview the dataset",
            "show the first 10 rows",
            "list candidates",
            "head of the data",
            "show preview of data"
        ]
    },
    {
        "intent": "single_numeric_aggregation",
        "description": "Calculate overall average, mean, sum, total, min, max of a numeric column without grouping",
        "examples": [
            "what is the average expected salary",
            "total expected salary of all candidates",
            "minimum experience in the dataset",
            "maximum interview score",
            "average years of experience"
        ]
    }
]

def get_sentence_transformer():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        # Load the lightweight model
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def get_template_embeddings():
    global _template_embeddings
    if _template_embeddings is None:
        model = get_sentence_transformer()
        # Flatten all examples and map back to intent
        flat_examples = []
        flat_intents = []
        for temp in TEMPLATES:
            for ex in temp["examples"]:
                flat_examples.append(ex)
                flat_intents.append(temp["intent"])
        
        embeddings = model.encode(flat_examples, convert_to_tensor=True)
        _template_embeddings = {
            "embeddings": embeddings,
            "intents": flat_intents,
            "examples": flat_examples
        }
    return _template_embeddings

def classify_intent_semantically(query: str) -> str:
    from sentence_transformers import util
    
    model = get_sentence_transformer()
    temp_data = get_template_embeddings()
    
    query_emb = model.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_emb, temp_data["embeddings"])[0]
    
    # Find the index of the highest similarity
    best_idx = int(similarities.argmax())
    best_score = float(similarities[best_idx])
    best_intent = temp_data["intents"][best_idx]
    
    print(f"Semantic Intent Match: {best_intent} (Score: {best_score:.4f}, Matched: '{temp_data['examples'][best_idx]}')")
    
    # If the match score is decent, return the matched intent, otherwise default
    if best_score > 0.35:
        return best_intent
    return "grouped_aggregation"

def find_semantic_column(query: str, columns: List[str], model) -> str:
    """Finds the column name in `columns` that is semantically closest to the query."""
    from sentence_transformers import util
    
    # If there's an exact string match (case-insensitive), prioritize it!
    query_lower = query.lower()
    for col in columns:
        col_lower = col.lower()
        if col_lower in query_lower or col_lower.replace('_', ' ') in query_lower:
            return col
            
    # Fallback to semantic similarity
    query_emb = model.encode(query, convert_to_tensor=True)
    col_embs = model.encode(columns, convert_to_tensor=True)
    similarities = util.cos_sim(query_emb, col_embs)[0]
    best_idx = int(similarities.argmax())
    return columns[best_idx]

def get_fallback_mock_response(query: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Provides a Semantic-Matcher model using Sentence-Transformers to map 
    the query to an intent template and extract the referenced columns semantically.
    """
    query_lower = query.lower()
    
    # 1. Column classification
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    # 2. Extract value filters from categorical columns
    filter_conditions = []
    filter_explanations = []
    
    for col in cat_cols:
        if df[col].nunique() < 50:
            for val in df[col].dropna().unique():
                val_str = str(val)
                pattern = r'\b' + re.escape(val_str.lower()) + r'\b'
                if re.search(pattern, query_lower):
                    filter_conditions.append(f"df['{col}'] == '{val}'")
                    filter_explanations.append(f"where {col} is '{val}'")
                    break
                    
    # 3. Classify intent and find referenced columns using local model
    try:
        model = get_sentence_transformer()
        intent = classify_intent_semantically(query)
        
        # Semantic column matching
        num_match = find_semantic_column(query, num_cols, model) if num_cols else None
        cat_match = find_semantic_column(query, cat_cols, model) if cat_cols else None
    except Exception as e:
        print(f"Warning: Failed to run local semantic matcher: {e}. Falling back to keyword matches.")
        # Fallback keyword matching
        intent = "grouped_aggregation"
        num_match = num_cols[0] if num_cols else None
        cat_match = cat_cols[0] if cat_cols else None
        model = None

    # Detect aggregation operation keywords
    is_avg = any(w in query_lower for w in ["average", "mean", "avg"])
    is_sum = any(w in query_lower for w in ["sum", "total", "add"])
    is_min = any(w in query_lower for w in ["min", "minimum", "lowest"])
    is_max = any(w in query_lower for w in ["max", "maximum", "highest", "greatest"])
    is_count = any(w in query_lower for w in ["count", "number of", "how many", "quantity"])
    
    code_lines = []
    explanation_parts = []
    
    # Base source
    df_source = "df"
    if filter_conditions:
        filter_expr = " & ".join(f"({cond})" for cond in filter_conditions)
        code_lines.append(f"filtered_df = df[{filter_expr}]")
        df_source = "filtered_df"
        explanation_parts.append("Filtered dataset " + " and ".join(filter_explanations))
        
    chart_type = "none"
    x_key = None
    y_keys = []
    
    # Execute mapping based on intent
    if intent == "correlation" and len(num_cols) >= 2:
        if model:
            # Rank numeric columns by similarity
            from sentence_transformers import util
            query_emb = model.encode(query, convert_to_tensor=True)
            col_embs = model.encode(num_cols, convert_to_tensor=True)
            similarities = util.cos_sim(query_emb, col_embs)[0]
            best_indices = similarities.argsort(descending=True)[:2].tolist()
            matched_nums = [num_cols[i] for i in best_indices]
        else:
            matched_nums = num_cols[:2]
            
        x_col = matched_nums[0]
        y_col = matched_nums[1]
        
        code_lines.append(f"result_df = {df_source}[['{x_col}', '{y_col}']].dropna()")
        explanation_parts.append(f"Compared '{x_col}' and '{y_col}' using a scatter plot")
        chart_type = "scatter"
        x_key = x_col
        y_keys = [y_col]
        
    elif intent == "grouped_aggregation" and cat_match and num_match:
        agg = "mean"
        agg_label = "average"
        if is_sum:
            agg = "sum"
            agg_label = "total"
        elif is_min:
            agg = "min"
            agg_label = "minimum"
        elif is_max:
            agg = "max"
            agg_label = "maximum"
            
        code_lines.append(f"result_df = {df_source}.groupby('{cat_match}')['{num_match}'].{agg}().reset_index()")
        explanation_parts.append(f"Calculated the {agg_label} of '{num_match}' grouped by '{cat_match}'")
        chart_type = "bar"
        x_key = cat_match
        y_keys = [num_match]
        
    elif intent == "grouped_count" and cat_match:
        code_lines.append(f"result_df = {df_source}['{cat_match}'].value_counts().reset_index()")
        code_lines.append(f"result_df.columns = ['{cat_match}', 'Count']")
        explanation_parts.append(f"Counted occurrences grouped by '{cat_match}'")
        
        unique_len = int(df[cat_match].nunique())
        chart_type = "pie" if unique_len < 7 else "bar"
        x_key = cat_match
        y_keys = ["Count"]
        
    elif intent == "single_numeric_aggregation" and num_match:
        agg = "mean"
        agg_label = "Average"
        if is_sum:
            agg = "sum"
            agg_label = "Total"
        elif is_min:
            agg = "min"
            agg_label = "Minimum"
        elif is_max:
            agg = "max"
            agg_label = "Maximum"
            
        code_lines.append(f"val = {df_source}['{num_match}'].{agg}()")
        code_lines.append(f"result_df = pd.DataFrame([{{'Metric': '{agg_label} {num_match}', 'Value': val}}])")
        explanation_parts.append(f"Calculated the overall {agg_label.lower()} of '{num_match}'")
        chart_type = "none"
        
    elif intent == "summary_statistics":
        code_lines.append(f"result_df = {df_source}.describe().reset_index()")
        explanation_parts.append("Generated descriptive summary statistics for numerical columns")
        chart_type = "none"
        
    elif intent == "preview_dataset":
        code_lines.append(f"result_df = {df_source}.head(10)")
        explanation_parts.append("Displaying the first 10 rows of the dataset")
        chart_type = "none"
        
    else:
        # Fallback
        if cat_cols and num_cols:
            x_col = cat_cols[0]
            y_col = num_cols[0]
            code_lines.append(f"result_df = {df_source}.groupby('{x_col}')['{y_col}'].mean().reset_index().head(10)")
            explanation_parts.append(f"Aggregated average '{y_col}' grouped by '{x_col}' as a default fallback")
            chart_type = "bar"
            x_key = x_col
            y_keys = [y_col]
        else:
            code_lines.append(f"result_df = {df_source}.head(10)")
            explanation_parts.append("Showing a preview of the dataset as a default fallback")
            chart_type = "none"
            
    code = "\n".join(code_lines)
    explanation = ". ".join(explanation_parts) + " (Local Semantic Matcher Model)."
            
    try:
        result_df = execute_pandas_code(code, df)
        chart_data = result_df.to_dict(orient="records")
        return {
            "explanation": explanation,
            "python_code": code,
            "chart_type": chart_type,
            "x_key": x_key,
            "y_keys": y_keys,
            "data": chart_data,
            "columns": list(result_df.columns),
            "success": True,
            "has_api_key": True
        }
    except Exception as e:
        return {
            "explanation": f"Failed to execute local query: {e}",
            "python_code": code,
            "chart_type": "none",
            "x_key": None,
            "y_keys": [],
            "data": [],
            "columns": [],
            "success": False,
            "error": str(e),
            "has_api_key": True
        }

def retrieve_relevant_file_semantically(query: str, upload_dir: str) -> Tuple[Optional[str], float]:
    """Uses the local Sentence-Transformer model to retrieve the most relevant 
    uploaded CSV file based on the query and file schemas.
    """
    from sentence_transformers import util
    
    # 1. List all CSV files in the upload directory
    csv_files = [f for f in os.listdir(upload_dir) if f.endswith(".csv")]
    if not csv_files:
        return None, 0.0
        
    # If there's only 1 file, return it directly with 100% confidence
    if len(csv_files) == 1:
        return csv_files[0], 1.0
        
    # 2. Build schema description for each file
    descriptions = []
    for f in csv_files:
        try:
            filepath = os.path.join(upload_dir, f)
            # Read only the header to be fast
            df_head = pd.read_csv(filepath, nrows=0)
            cols = list(df_head.columns)
            desc = f"File name: {f}. Column fields: {', '.join(cols)}"
            descriptions.append(desc)
        except Exception:
            descriptions.append(f"File name: {f}")
            
    # 3. Embed query and file descriptions
    try:
        model = get_sentence_transformer()
        query_emb = model.encode(query, convert_to_tensor=True)
        desc_embs = model.encode(descriptions, convert_to_tensor=True)
        
        # 4. Compute similarities
        similarities = util.cos_sim(query_emb, desc_embs)[0]
        best_idx = int(similarities.argmax())
        best_score = float(similarities[best_idx])
        best_file = csv_files[best_idx]
        
        print(f"RAG Schema Retrieval Match: {best_file} (Confidence: {best_score:.4f})")
        return best_file, best_score
    except Exception as e:
        print(f"Error in semantic file retrieval: {e}")
        return csv_files[0], 0.5

ROW_EMBEDDINGS_CACHE = {}

def run_rag_query(query: str, df: pd.DataFrame, filename: Optional[str] = None) -> Dict[str, Any]:
    global ROW_EMBEDDINGS_CACHE
    
    # 1. Prepare/retrieve row embeddings
    cache_key = filename or "temp_dataset"
    
    # Load SentenceTransformer model
    model = get_sentence_transformer()
    
    if cache_key not in ROW_EMBEDDINGS_CACHE:
        # Limit to first 2000 rows for performance
        max_rows = 2000
        df_slice = df.head(max_rows)
        
        row_texts = []
        for _, row in df_slice.iterrows():
            parts = []
            for col, val in row.items():
                if pd.notnull(val) and str(val).strip() != "":
                    parts.append(f"{col}: {val}")
            row_texts.append(" | ".join(parts))
            
        raw_rows = df_slice.replace([np.inf, -np.inf], np.nan).where(pd.notnull(df_slice), None).to_dict(orient="records")
        
        # Generate embeddings
        embeddings = model.encode(row_texts, convert_to_tensor=True, show_progress_bar=False)
        
        ROW_EMBEDDINGS_CACHE[cache_key] = {
            "embeddings": embeddings,
            "row_texts": row_texts,
            "raw_rows": raw_rows
        }
        
    cached_data = ROW_EMBEDDINGS_CACHE[cache_key]
    
    if not cached_data["row_texts"]:
        return {
            "explanation": "No valid data rows found in dataset to perform semantic search.",
            "python_code": "# RAG Mode - No rows to search",
            "chart_type": "none",
            "x_key": None,
            "y_keys": [],
            "data": [],
            "columns": list(df.columns),
            "success": False,
            "has_api_key": True,
            "provider": "local"
        }
        
    # 2. Compute similarity
    from sentence_transformers import util
    query_emb = model.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_emb, cached_data["embeddings"])[0]
    
    # 3. Retrieve top K matches
    top_k = min(10, len(cached_data["row_texts"]))
    top_k_indices = similarities.argsort(descending=True)[:top_k].tolist()
    matched_rows = [cached_data["raw_rows"][idx] for idx in top_k_indices]
    
    # Format retrieved context
    context_lines = []
    for i, idx in enumerate(top_k_indices):
        score = float(similarities[idx])
        row_text = cached_data["row_texts"][idx]
        context_lines.append(f"Match #{i+1} (Similarity Score: {score:.2f}):\n{row_text}")
    context_str = "\n\n".join(context_lines)
    
    # 4. Check model provider settings
    app_settings = load_settings()
    provider = app_settings.provider
    
    if provider == "local":
        # Local semantic search response (no external LLM key needed)
        explanation = f"Returned the top {top_k} semantically matching records from the dataset (Local Semantic Search):\n\n"
        for i, idx in enumerate(top_k_indices):
            score = float(similarities[idx])
            explanation += f"- **Match #{i+1} (Similarity: {score:.2f})**: {cached_data['row_texts'][idx][:150]}...\n"
        return {
            "explanation": explanation,
            "python_code": "# Local Semantic Search",
            "chart_type": "none",
            "x_key": None,
            "y_keys": [],
            "data": matched_rows,
            "columns": list(df.columns),
            "success": True,
            "has_api_key": False,
            "provider": "local"
        }
        
    # Build prompt for LLM provider
    prompt = f"""You are an expert Data Analyst assisting a user.
A user has asked a question about a CSV dataset:
"{query}"

We performed a semantic similarity search and retrieved the most relevant rows from the CSV file.
Here are the retrieved records:
{context_str}

### Instructions:
1. Synthesize a clear, accurate, and concise answer to the user's question using ONLY the retrieved records above.
2. If the retrieved records do not contain enough information to answer the question, state that clearly in your answer.
3. You must respond with a JSON object matching this schema:
{{
  "explanation": "Your detailed answer based on the retrieved rows.",
  "python_code": "# RAG Mode - Rows retrieved using Sentence-Transformers",
  "chart_type": "none",
  "x_key": null,
  "y_keys": []
}}
"""

    try:
        parsed_response = None
        
        if provider == "gemini":
            api_key = app_settings.gemini_api_key
            if not api_key:
                raise ValueError("Gemini API Key not found. Please add it in Settings.")
            genai.configure(api_key=api_key)
            model_name = app_settings.gemini_model or "gemini-3.5-flash"
            model_instance = genai.GenerativeModel(model_name)
            
            response = model_instance.generate_content(
                prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": QueryResponseSchema,
                }
            )
            parsed_response = json.loads(response.text)
            
        elif provider == "openai":
            api_key = app_settings.openai_api_key
            if not api_key:
                raise ValueError("OpenAI API Key not found. Please add it in Settings.")
            client = OpenAI(api_key=api_key)
            model_name = app_settings.openai_model or "gpt-4o-mini"
            
            response = client.beta.chat.completions.parse(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                response_format=QueryResponseSchema,
            )
            parsed_model = response.choices[0].message.parsed
            if parsed_model:
                parsed_response = parsed_model.model_dump()
            else:
                raw_content = response.choices[0].message.content or ""
                parsed_response = json.loads(raw_content)
                
        elif provider == "anthropic":
            api_key = app_settings.anthropic_api_key
            if not api_key:
                raise ValueError("Anthropic API Key not found. Please add it in Settings.")
            client = anthropic.Anthropic(api_key=api_key)
            model_name = app_settings.anthropic_model or "claude-3-5-sonnet-latest"
            
            response = client.messages.create(
                model=model_name,
                max_tokens=4000,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                tools=[
                    {
                        "name": "respond_with_analysis",
                        "description": "Respond with the structured analysis output.",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "explanation": {"type": "string", "description": "Explanation of the data analysis approach and result."},
                                "python_code": {"type": "string", "description": "Executable Python code."},
                                "chart_type": {"type": "string", "enum": ["bar", "line", "pie", "scatter", "none"], "description": "Type of chart to display."},
                                "x_key": {"type": "string", "description": "Column name for the X-axis of the chart (null if none)."},
                                "y_keys": {"type": "array", "items": {"type": "string"}, "description": "List of column names for the Y-axis / values."}
                            },
                            "required": ["explanation", "python_code", "chart_type"]
                        }
                    }
                ],
                tool_choice={"type": "tool", "name": "respond_with_analysis"}
            )
            
            tool_use = next(block for block in response.content if block.type == "tool_use")
            parsed_response = tool_use.input
            
        elif provider == "ollama":
            api_base = app_settings.ollama_api_base or "http://localhost:11434/v1"
            model_name = app_settings.ollama_model or "llama3"
            
            client = OpenAI(base_url=api_base, api_key="ollama")
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            raw_content = response.choices[0].message.content or ""
            parsed_response = json.loads(raw_content)
            
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
            
        if not parsed_response:
            raise ValueError("No response received from LLM provider.")
            
        explanation = parsed_response.get("explanation") or ""
        code_to_run = parsed_response.get("python_code") or parsed_response.get("pythonCode") or "# RAG Mode"
        
        return {
            "explanation": explanation,
            "python_code": code_to_run,
            "chart_type": "none",
            "x_key": None,
            "y_keys": [],
            "data": matched_rows,
            "columns": list(df.columns),
            "success": True,
            "has_api_key": True,
            "provider": provider
        }
        
    except Exception as e:
        print(f"Error in RAG execution ({provider}): {e}")
        # Fall back to local semantic search on error
        local_fallback = {
            "explanation": f"Warning: Provider '{provider}' failed ({e}). Fell back to Local Semantic Search.\n\n" + 
                           f"Returned the top {top_k} semantically matching records from the dataset:\n\n" +
                           "\n".join(f"- **Match #{i+1} (Similarity: {similarities[idx]:.2f})**: {cached_data['row_texts'][idx][:150]}..." for i, idx in enumerate(top_k_indices)),
            "python_code": "# Local Semantic Search fallback",
            "chart_type": "none",
            "x_key": None,
            "y_keys": [],
            "data": matched_rows,
            "columns": list(df.columns),
            "success": True,
            "has_api_key": False,
            "provider": "local"
        }
        return local_fallback
