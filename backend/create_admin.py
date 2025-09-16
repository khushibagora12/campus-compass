# backend/create_admin.py
import getpass
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from supabase_client import supabase
from auth_utils import hash_password

def create_admin_interactive():
    """Interactively prompts for new admin details and creates the user."""
    print("--- Create a New Campus Compass Admin ---")
    
    try:
        username = input("Enter new admin username (e.g., 'sgsits_admin'): ").strip()
        password = getpass.getpass("Enter a secure password for this admin: ").strip()
        college_id = input(f"Enter the college_id for '{username}' (e.g., 'sgsits'): ").strip()

        if not all([username, password, college_id]):
            print("\nError: All fields are required. Aborting.")
            return

        # Check if user already exists
        response = supabase.table('admins').select('id').eq('username', username).execute()
        
        if response.data:
            print(f"\nError: User '{username}' already exists.")
            return

        # Hash the password and insert the new admin
        hashed_pwd = hash_password(password)
        data, error = supabase.table('admins').insert({
            "username": username,
            "hashed_password": hashed_pwd,
            "college_id": college_id
        }).execute()
        
        # The Supabase python client returns data and error in a tuple
        # We need to check the second element of the tuple for the error.
        if error and error[1]:
            print(f"\nError creating admin: {error[1]}")
        else:
            print(f"\n✅ Successfully created admin user: '{username}' for college '{college_id}'")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    create_admin_interactive()