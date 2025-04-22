import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Sample rows from a ratings CSV.")
    parser.add_argument("--input_file", type=str, default="ratings.csv",
                        help="Path to the input CSV file (userId,movieId,rating,timestamp).")
    parser.add_argument("--n_rows", type=int, default=10,
                        help="Number of rows to sample.")
    parser.add_argument("--output_test", type=str, default="sample_test.csv",
                        help="Output file for userId,movieId.")
    parser.add_argument("--output_test_with_ratings", type=str, default="sample_test_with_ratings.csv",
                        help="Output file for userId,movieId,rating.")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducible sampling.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Read the input CSV
    df = pd.read_csv(args.input_file)  # columns: userId,movieId,rating,timestamp

    # Randomly sample n_rows from the CSV
    df_sample = df.sample(n=args.n_rows, random_state=args.random_seed)

    # Create a DataFrame for sample_test.csv (only userId,movieId)
    df_test = df_sample[["userId", "movieId"]].copy()

    # Create a DataFrame for sample_test_with_ratings.csv (userId,movieId,rating)
    df_test_with_ratings = df_sample[["userId", "movieId", "rating"]].copy()

    # Save the output files
    df_test.to_csv(args.output_test, index=False)
    df_test_with_ratings.to_csv(args.output_test_with_ratings, index=False)

    print(f"Sampled {args.n_rows} rows from {args.input_file}.")
    print(f"Created '{args.output_test}' with userId,movieId.")
    print(f"Created '{args.output_test_with_ratings}' with userId,movieId,rating.")

if __name__ == "__main__":
    main()
