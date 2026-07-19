import pandas as pd
  # public nb #1
sub1 = pd.read_csv("/Users/abdullah/Desktop/Kaggle comp/submission_1.csv")  # public nb #2
sub2 = pd.read_csv("/Users/abdullah/Desktop/Kaggle comp/submission_2.csv")  # public nb #3
sub3 = pd.read_csv("/Users/abdullah/Desktop/Kaggle comp/submission_3.csv")  # your V2
sub4 = pd.read_csv("/Users/abdullah/Desktop/Kaggle comp/submission_4.csv")  # your V3
sub5 = pd.read_csv("/Users/abdullah/Desktop/Kaggle comp/submission_5.csv")  # your V4
sub6 = pd.read_csv("/Users/abdullah/Desktop/Kaggle comp/submission_6.csv")  # your V5
sub7 = pd.read_csv("/Users/abdullah/Desktop/Kaggle comp/submission_7.csv")  # your V6
sub8 = pd.read_csv("/Users/abdullah/Desktop/Kaggle comp/submission_8.csv")  # your V7
sub9 = pd.read_csv("/Users/abdullah/Desktop/Kaggle comp/submission_9.csv")  # your V8
sub10 = pd.read_csv("/Users/abdullah/Desktop/Kaggle comp/submission_10.csv")  # your V9
sub11 = pd.read_csv("/Users/abdullah/Desktop/Kaggle comp/submission_11.csv")  # your V10
sub12 = pd.read_csv("/Users/abdullah/Desktop/Kaggle comp/submission_12.csv")  # your V11
mine = pd.read_csv("/Users/abdullah/Desktop/Kaggle comp/submission_pytorch.csv")


blend = sub1.copy()
blend["PitNextLap"] = (
    sub1["PitNextLap"] + 
    sub2["PitNextLap"] + 
    sub3["PitNextLap"] + 
    mine["PitNextLap"] +
    sub4["PitNextLap"] +
    sub5["PitNextLap"] +
    sub6["PitNextLap"] +
    sub7["PitNextLap"] +
    sub8["PitNextLap"] +
    sub9["PitNextLap"] +
    sub10["PitNextLap"] +
    sub11["PitNextLap"] +
    sub12["PitNextLap"] 
) / 13

blend.to_csv("/Users/abdullah/Desktop/Kaggle comp/submission.csv", index=False)



