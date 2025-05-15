import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def add_new_user(user_id, tags, users, encoder):
    new_user = pd.DataFrame([tags], columns=users.columns[1:])
    new_user.insert(0, "user_id", user_id)
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv("users.csv", index=False)
    return users

def add_interaction(user_id, course_id, liked, interactions):
    new_interaction = pd.DataFrame([[user_id, course_id, liked]], columns=["user_id", "course_id", "liked"])
    interactions = pd.concat([interactions, new_interaction], ignore_index=True)
    interactions.to_csv("interactions.csv", index=False)
    return interactions