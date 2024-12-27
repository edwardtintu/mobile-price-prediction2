import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Open file dialog to select the CSV file
Tk().withdraw()  # Prevents the root window from appearing
file_path = askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")])

if file_path:
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Get the first 5 rows and convert them to JSON
    first_five_rows_json = df.head(5).to_json(orient='records')
    print(first_five_rows_json)
else:
    print("No file selected.")
