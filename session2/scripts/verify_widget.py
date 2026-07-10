import os

def main():
    widget_dir = "/Users/kamran/Documents/ERAV5/session2/widget"
    required_files = [
        "index.html",
        "app.js",
        "tokenizer_data.js",
        "sample_texts.js"
    ]
    
    print("Verifying widget assets...")
    missing_files = []
    for filename in required_files:
        filepath = os.path.join(widget_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f" - {filename:<20}: Present ({size} bytes)")
        else:
            print(f" - {filename:<20}: MISSING!")
            missing_files.append(filename)
            
    if missing_files:
        print("\nVerification: FAIL!")
        print(f"Missing files: {missing_files}")
    else:
        print("\nVerification: SUCCESS!")
        print("All widget assets are ready for submission / deployment.")

if __name__ == "__main__":
    main()
