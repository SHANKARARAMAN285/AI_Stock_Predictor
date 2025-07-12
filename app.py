import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime

# --- Page Config ---
st.set_page_config(page_title="Stock Market Predictor", layout="centered")

# --- CSS Styling ---
st.markdown("""
    <style>
        .main { background-color: #f0f2f6; padding: 20px; border-radius: 10px; }
        h1, h2, h3 { color: #1f77b4; }
        .stButton>button { background-color: #1f77b4; color: white; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# --- Credentials ---
USERNAME = "user123"
PASSWORD = "password123"

# --- Session State ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# --- Login Page ---
def login_page():
    st.title("🔐 Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.success("Login successful!")
        else:
            st.error("Incorrect username or password")

# --- Prediction Page ---
def prediction_page():
    st.title("📈 AI-Based Stock Market Predictor")

    tab1, tab2, tab3 = st.tabs(["🔍 Stock Info & Predict", "📊 Prediction Graph", "ℹ About Project"])

    with tab1:
        st.subheader("Enter Stock Details")

        stock = st.text_input("Enter Stock Symbol (e.g., TCS.NS)", "TCS.NS")
        start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
        end_date = st.date_input("End Date", pd.to_datetime("2024-01-01"))
        
        # Prevent future dates for end date
        if end_date > datetime.today().date():
            st.warning("⚠ You cannot predict future stock data beyond today's date.")
            end_date = datetime.today().date()
        
        predict_btn = st.button("Predict")

        if predict_btn and stock:
            try:
                # Fetch stock data within the user-selected date range
                df = yf.download(stock, start=start_date, end=end_date)
                
                if df.empty:
                    st.warning("⚠ No data found. Try another stock symbol.")
                    return

                # Show Company Info
                st.subheader("🏢 Company Info")
                info = yf.Ticker(stock).info
                st.write(f"*Name:* {info.get('longName', 'N/A')}")
                st.write(f"*Sector:* {info.get('sector', 'N/A')}")
                st.write(f"*Industry:* {info.get('industry', 'N/A')}")
                st.write(f"*Market Cap:* {info.get('marketCap', 'N/A')}")
                st.write(f"*Current Price:* {info.get('currentPrice', 'N/A')}")

                # Prepare data for ML (Linear Regression)
                df = df[['Close']]
                df['Tomorrow'] = df['Close'].shift(-1)
                df.dropna(inplace=True)

                X = df[['Close']].values
                y = df['Tomorrow'].values

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                model = LinearRegression()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                # Save prediction results to session
                st.session_state.df_result = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': predictions
                })

                st.session_state.df_result['Error %'] = np.abs(
                    (st.session_state.df_result['Actual'] - st.session_state.df_result['Predicted']) / st.session_state.df_result['Actual']) * 100

                st.success("✅ Prediction complete! View the graph in the next tab.")

            except Exception as e:
                st.error(f"An error occurred: {e}")

    with tab2:
        st.subheader("📊 Prediction Graph")
        
        if "df_result" in st.session_state:
            df = st.session_state.df_result

            # Plot the prediction graph
            plt.figure(figsize=(10, 5))
            plt.plot(df['Actual'].values, color='red', label='Actual')
            plt.plot(df['Predicted'].values, color='blue', label='Predicted')
            plt.title("Actual vs Predicted Stock Prices")
            plt.xlabel("Days")
            plt.ylabel("Price")
            plt.legend()
            st.pyplot(plt)

            # Prediction Table (with selected date range)
            st.write("📋 Prediction Table (based on the selected date range)")
            
            # Add corresponding dates to the table (using df.index)
            prediction_table = pd.DataFrame({
                'Date': df.index,
                'Actual': df['Actual'],
                'Predicted': df['Predicted'],
                'Error %': df['Error %']
            })

            st.dataframe(prediction_table)
        else:
            st.info("ℹ Run a prediction in the first tab to see the graph and table here.")

    with tab3:
        st.header("ℹ About This Project")
        st.markdown("""
        - 📚 *Project Title:* Stock Market Predictor using AI  
        - 👨‍💻 *Developer:* Your Name  
        - 🏫 *For:* 2nd Year B.Tech CSE Mini Project  
        - 🧠 *Tech Used:* Streamlit, yfinance, scikit-learn, matplotlib  
        - 🎯 *Goal:* Predict next-day stock price using simple AI model  
        - 🗓 *Submitted On:* April 25, 2025
        """)

    st.markdown("---")
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()

# --- Main App ---
def main():
    if not st.session_state.logged_in:
        login_page()
    else:
        prediction_page()

main()
