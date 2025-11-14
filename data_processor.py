"""
Banking Data Processor
Handles loading and processing of banking data for the chatbot
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import os


class BankingDataProcessor:
    """Processes banking data for the RAG system"""
    
    def __init__(self, 
                 accounts_path: str,
                 transactions_path: str, 
                 users_path: str,
                 user_financials_path: str):
        """
        Initialize data processor with file paths
        
        Args:
            accounts_path: Path to accounts CSV
            transactions_path: Path to transactions CSV  
            users_path: Path to users CSV
            user_financials_path: Path to user financials CSV
        """
        self.accounts_path = accounts_path
        self.transactions_path = transactions_path
        self.users_path = users_path
        self.user_financials_path = user_financials_path
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all data files"""
        data = {}
        
        try:
            data['accounts'] = pd.read_csv(self.accounts_path)
            data['transactions'] = pd.read_csv(self.transactions_path)
            data['users'] = pd.read_csv(self.users_path)
            data['user_financials'] = pd.read_csv(self.user_financials_path)
            
            print("✅ All data files loaded successfully")
            return data
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def preprocess_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Preprocess and clean the data"""
        
        # Normalize and clean accounts data
        if 'accounts' in data:
            # Normalize possible column names to expected internal names
            acc = data['accounts']
            # Map variations to standard names
            if 'account_number' not in acc.columns:
                if 'accountNo' in acc.columns:
                    acc['account_number'] = acc['accountNo']
                elif 'accountno' in [c.lower() for c in acc.columns]:
                    # fallback for oddly-cased names
                    for c in acc.columns:
                        if c.lower() == 'accountno':
                            acc['account_number'] = acc[c]
                            break
            if 'user_id' not in acc.columns:
                if 'userId' in acc.columns:
                    acc['user_id'] = acc['userId']
                elif 'userid' in [c.lower() for c in acc.columns]:
                    for c in acc.columns:
                        if c.lower() == 'userid':
                            acc['user_id'] = acc[c]
                            break
            if 'account_type' not in acc.columns:
                if 'accountType' in acc.columns:
                    acc['account_type'] = acc['accountType']
                elif 'accounttype' in [c.lower() for c in acc.columns]:
                    for c in acc.columns:
                        if c.lower() == 'accounttype':
                            acc['account_type'] = acc[c]
                            break

            # Balance numeric cleanup
            if 'balance' in acc.columns:
                acc['balance'] = pd.to_numeric(acc['balance'], errors='coerce').fillna(0)
            else:
                acc['balance'] = 0.0

            # Fill defaults for optional columns
            if 'currency' not in acc.columns:
                acc['currency'] = 'USD'
            if 'status' not in acc.columns:
                acc['status'] = 'Active'

            data['accounts'] = acc
            
        # Normalize and clean transactions data
        if 'transactions' in data:
            txn = data['transactions']

            # Normalize account column: prefer 'account_number', else try obvious alternatives
            if 'account_number' not in txn.columns:
                if 'accountNo' in txn.columns:
                    txn['account_number'] = txn['accountNo']
                elif 'toAccountNo' in txn.columns:
                    txn['account_number'] = txn['toAccountNo']
                elif 'fromAccountNo' in txn.columns:
                    txn['account_number'] = txn['fromAccountNo']
                else:
                    # try case-insensitive match
                    for c in txn.columns:
                        if c.lower() in ('accountno', 'toaccountno', 'fromaccountno'):
                            txn['account_number'] = txn[c]
                            break

            # Normalize user id
            if 'user_id' not in txn.columns:
                if 'userId' in txn.columns:
                    txn['user_id'] = txn['userId']
                elif 'userid' in [c.lower() for c in txn.columns]:
                    for c in txn.columns:
                        if c.lower() == 'userid':
                            txn['user_id'] = txn[c]
                            break

            # Normalize transaction id
            if 'transaction_id' not in txn.columns:
                if 'id' in txn.columns:
                    txn['transaction_id'] = txn['id']

            # Normalize transaction type
            if 'transaction_type' not in txn.columns:
                if 'transaction_type' in txn.columns:
                    pass
                elif 'type' in txn.columns:
                    txn['transaction_type'] = txn['type']
                elif 'transactionType' in txn.columns:
                    txn['transaction_type'] = txn['transactionType']

            # Amount numeric cleanup
            if 'amount' in txn.columns:
                txn['amount'] = pd.to_numeric(txn['amount'], errors='coerce').fillna(0)
            else:
                txn['amount'] = 0.0

            # Date normalization
            date_candidates = ['transaction_date', 'date', 'posted_date']
            for dc in date_candidates:
                if dc in txn.columns:
                    txn['transaction_date'] = pd.to_datetime(txn[dc], errors='coerce')
                    break
            if 'transaction_date' not in txn.columns:
                txn['transaction_date'] = pd.NaT

            data['transactions'] = txn
        
        # Clean user financials
        if 'user_financials' in data:
            numeric_columns = ['monthly_income', 'credit_score', 'total_debt']
            for col in numeric_columns:
                if col in data['user_financials'].columns:
                    data['user_financials'][col] = pd.to_numeric(
                        data['user_financials'][col], errors='coerce'
                    ).fillna(0)

            # Normalize common variations
            uf = data['user_financials']
            if 'user_id' not in uf.columns:
                if 'userId' in uf.columns:
                    uf['user_id'] = uf['userId']
                elif 'id' in uf.columns:
                    uf['user_id'] = uf['id']

            if 'monthly_income' not in uf.columns:
                if 'monthlyIncome' in uf.columns:
                    uf['monthly_income'] = pd.to_numeric(uf['monthlyIncome'], errors='coerce').fillna(0)

            if 'credit_score' not in uf.columns:
                if 'creditScore' in uf.columns:
                    uf['credit_score'] = pd.to_numeric(uf['creditScore'], errors='coerce').fillna(0)

            # Defaults
            if 'total_debt' not in uf.columns:
                uf['total_debt'] = 0.0
            if 'risk_tolerance' not in uf.columns:
                # try to map from spendingCategory if present
                if 'spendingCategory' in uf.columns:
                    uf['risk_tolerance'] = uf['spendingCategory']
                else:
                    uf['risk_tolerance'] = 'Unknown'

            if 'financial_goals' not in uf.columns:
                uf['financial_goals'] = ''

            data['user_financials'] = uf
        
        return data
    
    def create_documents(self, data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Create documents for vector store from processed data"""
        documents = []
        
        # Create account documents
        for _, account in data['accounts'].iterrows():
            doc = {
                'content': f"""
                Account Information:
                - Account ID: {account['account_number']}
                - Type: {account['account_type']}
                - Balance: ${account['balance']:,.2f}
                - Currency: {account['currency']}
                - Status: {account['status']}
                - User ID: {account['user_id']}
                """.strip(),
                'metadata': {
                    'type': 'account',
                    'user_id': account['user_id'],
                    'account_number': account['account_number'],
                    'balance': account['balance'],
                    'source': 'accounts'
                }
            }
            documents.append(doc)
        
        # Create transaction documents (recent ones)
        recent_transactions = data['transactions'].sort_values('transaction_date', ascending=False).head(100)
        for _, transaction in recent_transactions.iterrows():
            doc = {
                'content': f"""
                Transaction Record:
                - Transaction ID: {transaction['transaction_id']}
                - Account: {transaction['account_number']}
                - Amount: ${transaction['amount']:,.2f}
                - Type: {transaction['transaction_type']}
                - Description: {transaction['description']}
                - Date: {transaction['transaction_date']}
                - User ID: {transaction['user_id']}
                """.strip(),
                'metadata': {
                    'type': 'transaction',
                    'user_id': transaction['user_id'],
                    'account_number': transaction['account_number'],
                    'amount': transaction['amount'],
                    'date': str(transaction['transaction_date']),
                    'source': 'transactions'
                }
            }
            documents.append(doc)
        
        # Create user financial profile documents
        for _, financial in data['user_financials'].iterrows():
            doc = {
                'content': f"""
                User Financial Profile:
                - User ID: {financial['user_id']}
                - Monthly Income: ${financial['monthly_income']:,.2f}
                - Credit Score: {financial['credit_score']}
                - Total Debt: ${financial['total_debt']:,.2f}
                - Risk Tolerance: {financial['risk_tolerance']}
                - Financial Goals: {financial['financial_goals']}
                """.strip(),
                'metadata': {
                    'type': 'financial_profile',
                    'user_id': financial['user_id'],
                    'monthly_income': financial['monthly_income'],
                    'credit_score': financial['credit_score'],
                    'source': 'user_financials'
                }
            }
            documents.append(doc)
        
        # Create user information documents
        for _, user in data['users'].iterrows():
            status = "Active" if user['isActive'] == 1 else "Inactive"
            doc = {
                'content': f"""
                User Information:
                - User ID: {user['id']}
                - Username: {user['username']}
                - Email: {user['email']}
                - Role: {user['role']}
                - Status: {status}
                """.strip(),
                'metadata': {
                    'type': 'user_info',
                    'user_id': user['id'],
                    'username': user['username'],
                    'role': user['role'],
                    'source': 'users'
                }
            }
            documents.append(doc)
        
        print(f"✅ Created {len(documents)} documents for vector store")
        return documents
    
    def process_all(self) -> List[Dict[str, Any]]:
        """Complete data processing pipeline"""
        print("🔄 Starting data processing...")
        
        # Load data
        data = self.load_data()
        
        # Preprocess data
        data = self.preprocess_data(data)
        
        # Create documents
        documents = self.create_documents(data)
        
        print("✅ Data processing completed successfully")
        return documents


# Sample data creation function for testing
def create_sample_data():
    """Create sample banking data for testing"""
    
    # Sample accounts data
    accounts_data = {
        'account_number': ['CUR3342331444', 'SAV4456123789', 'CUR7788990011', 'INV9922334455'],
        'user_id': [1, 1, 2, 2],
        'account_type': ['Current', 'Savings', 'Current', 'Investment'],
        'balance': [59049.25, 125000.00, 23456.78, 150000.00],
        'currency': ['USD', 'USD', 'USD', 'USD'],
        'status': ['Active', 'Active', 'Active', 'Active']
    }
    
    # Sample transactions data
    transactions_data = {
        'transaction_id': ['TXN001', 'TXN002', 'TXN003', 'TXN004', 'TXN005'],
        'user_id': [1, 1, 2, 1, 2],
        'account_number': ['CUR3342331444', 'CUR3342331444', 'CUR7788990011', 'SAV4456123789', 'INV9922334455'],
        'amount': [1500.00, -250.50, 3000.00, -100.00, 500.00],
        'transaction_type': ['Credit', 'Debit', 'Credit', 'Debit', 'Credit'],
        'description': ['Salary Deposit', 'ATM Withdrawal', 'Business Payment', 'Monthly Fee', 'Dividend'],
        'transaction_date': ['2024-01-15', '2024-01-16', '2024-01-14', '2024-01-10', '2024-01-12']
    }
    
    # Sample users data
    users_data = {
        'id': [1, 2, 3],
        'username': ['john_doe', 'sarah_smith', 'mike_johnson'],
        'email': ['john.doe@email.com', 'sarah.smith@email.com', 'mike.johnson@email.com'],
        'role': ['Premium', 'Standard', 'Basic'],
        'isActive': [1, 1, 0]
    }
    
    # Sample user financials data
    user_financials_data = {
        'user_id': [1, 2, 3],
        'monthly_income': [7500.00, 5500.00, 4200.00],
        'credit_score': [780, 650, 720],
        'total_debt': [15000.00, 28000.00, 12000.00],
        'risk_tolerance': ['Medium', 'Low', 'High'],
        'financial_goals': ['Retirement Planning', 'Home Purchase', 'Debt Reduction']
    }
    
    # Create DataFrames and save to CSV
    os.makedirs('uploads', exist_ok=True)
    
    pd.DataFrame(accounts_data).to_csv('uploads/accounts.csv', index=False)
    pd.DataFrame(transactions_data).to_csv('uploads/transactions.csv', index=False)
    pd.DataFrame(users_data).to_csv('uploads/users.csv', index=False)
    pd.DataFrame(user_financials_data).to_csv('uploads/user_financials.csv', index=False)
    
    print("✅ Sample data created successfully")


if __name__ == "__main__":
    create_sample_data()