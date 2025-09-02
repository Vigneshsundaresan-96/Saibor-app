import streamlit as st
import pandas as pd
from saibor_platform import run_engine  # import your logic

st.title("SAIBOR Testing Platform")

uploaded_file = st.file_uploader("Upload Deals Excel (SAIBOR_Template.xlsx)", type="xlsx")
reporting_date = st.date_input("Reporting Date")

if st.button("Run Calculation"):
    if uploaded_file is not None:
        out_file = "SAIBOR_Result.xlsx"
        run_engine(uploaded_file, str(reporting_date), out_file)
        with open(out_file, "rb") as f:
            st.download_button(
                label="Download Results",
                data=f,
                file_name=out_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.warning("Please upload an Excel file first.")
