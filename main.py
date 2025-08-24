from Src.popularity import top_popular
from Src.Collaborative import collaborative_recommend
from Src.Data import load_ratings

def main():
    print("🎬 Movie Recommendation System")
    print("1. Popularity-based")
    print("2. Collaborative Filtering")

    choice = input("choose method (1/2): ")

    if choice == "1":
        print("\nTop Popular Movies:")
        print(top_popular(10, 1000))

    elif choice == "2":
        ratings = load_ratings().sample(n=50000, random_state=42)  # only once!
        user_id = ratings["userId"].sample(1).iloc[0]  # pick valid user
        print(f"\nRecommendations for User {user_id}:")
        print(collaborative_recommend(user_id, 10, ratings))  # pass ratings here

    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
