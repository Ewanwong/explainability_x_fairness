import argparse

# receive arguments from command line and print them
def return_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--lr", type=float, required=True, help="Dataset name (e.g., jigsaw, civil)")
    args = parser.parse_args()
    print(f"Learning Rate: {args.lr}")
    return args

if __name__ == "__main__":
    args = return_args()
    # You can use args.lr in your code as needed
    # For example, you can print it or use it in a function
    print(f"Received learning rate: {args.lr}")
    