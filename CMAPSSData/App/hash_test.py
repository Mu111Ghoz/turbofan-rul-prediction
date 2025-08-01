import streamlit_authenticator as stauth

# Plaintext passwords
passwords = ['engineerpass', 'analystpass']

# Try to hash
hashed_passwords = stauth.Hasher(passwords).generate()

print(hashed_passwords)
