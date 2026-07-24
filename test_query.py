import pandas as pd
import numpy as np
from query_engine import execute_pandas_code, run_insight_query

def run_tests():
    print("Starting tests...")
    
    # 1. Create a dummy dataframe
    data = {
        "Department": ["Sales", "Engineering", "Sales", "HR", "Engineering", "Engineering"],
        "Salary": [50000, 80000, 60000, 45000, 95000, 90000],
        "Age": [28, 34, 30, 29, 42, 38]
    }
    df = pd.DataFrame(data)
    print("Mock DataFrame created:")
    print(df)
    
    # 2. Test execute_pandas_code
    print("\nTesting execute_pandas_code...")
    code_str = """
result_df = df.groupby('Department')['Salary'].mean().reset_index()
"""
    result = execute_pandas_code(code_str, df)
    print("Result of code execution:")
    print(result)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert "Salary" in result.columns
    assert "Department" in result.columns
    print("[PASSED] execute_pandas_code passed!")
    
    # 3. Test run_insight_query (under fallback or real model)
    print("\nTesting run_insight_query...")
    query = "What is the average salary by department?"
    res = run_insight_query(query, df)
    
    print("Keys in response:", res.keys())
    assert "explanation" in res
    assert "python_code" in res
    assert "chart_type" in res
    assert "data" in res
    assert res["success"] is True
    print("[PASSED] run_insight_query passed!")
    
    # 4. Test run_insight_query in RAG mode (Local fallback)
    print("\nTesting run_insight_query in RAG mode...")
    rag_query = "Engineering with high salary"
    res_rag = run_insight_query(rag_query, df, mode="rag", filename="test_dataset")
    print("RAG Keys in response:", res_rag.keys())
    print("RAG Explanation:\n", res_rag["explanation"])
    print("RAG Data (first row):", res_rag["data"][0])
    
    assert "explanation" in res_rag
    assert "python_code" in res_rag
    assert "chart_type" in res_rag
    assert "data" in res_rag
    assert len(res_rag["data"]) > 0
    assert res_rag["success"] is True
    print("[PASSED] RAG mode query passed!")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    run_tests()
