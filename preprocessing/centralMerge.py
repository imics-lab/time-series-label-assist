import pandas as pd
import split

def validate(frame1, frame2):
    if frame1.columns == frame2.columns:
        return True
    else:
        return False

def vertMerge(frame1, frame2):
    if validate(frame1, frame2):
        return concat(frame1, frame2)
    else:
        print("failed to merge because of column misalignment")

def mergeHorizontal(df1, df2):
    return merge( df2, df1, how = 'outer'  ,on='date_time')