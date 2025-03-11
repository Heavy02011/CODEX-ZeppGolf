import sqlite3
import pandas as pd
import pytz

# Build your `wrangle` function here
def wrangle(db_path):
    # Connect to database
    conn = sqlite3.connect(db_path)

    # Construct query
    query = """
    SELECT UPSWING_CLUB_POSTURE, UP_DOWN_SWING__GOF, TWIST_ROTATION_RATE,
    IMPACT_SPEED, CLUB_FACE_GESTURE__GOF, ENDSWING_CLUB_POSTURE,
    UPSWING__A_TIME, UPSWING__B_TIME, TOP_HOLDING_TIME, TWIST_TIME,
    DOWNSWING_IMPACT_TIME, ENDSWING_TIME,
    FIRST_HALF_ANIMATION_END_FRAME,
    FIRST_HALF_ANIMATION_SAMPLE_POINT_NUMBER,
    SECOND_HALF_ANIMATION_START_FRAME, 
    SECOND_HALF_ANIMATION_SAMPLE_POINT_NUMBER, 
    BACK_SWING_TEMPO_SLOW, TRANSITION_TEMPO_FAST,
    HAND_SPEED, IMPACT_DETECT,
    HAND_FIT, CLUB_PLANE, HAND_PLANE, _ID, L_ID, S_ID,
    CLIENT_CREATED, CLUB_TYPE_1,
    CLUB_TYPE_2, CLUB_SHAFT_1, CLUB_SHAFT_2, CLUB_LENGTH,
    CLUB_POSTURE, CLUB_POSITION, USER_HEIGHT, 
    YEAR, MONTH, DAY, FACE_ANGLE, 
    SCORE, MODEL_ID, CLIENT_HOUR
    FROM swings
    """

    # Read query results into DataFrame
    # df = pd.read_sql(query, conn, index_col="time")
    df = pd.read_sql(query, conn)
    df = df.sort_index()  
    df = df.drop_duplicates()

    df['CLIENT_CREATED'] = pd.to_datetime(df['CLIENT_CREATED'], unit='ms')
    az_timezone = pytz.timezone('America/Phoenix')
    df['CLIENT_CREATED'] = df['CLIENT_CREATED'].dt.tz_localize('UTC').dt.tz_convert(az_timezone)
    df['CLIENT_CREATED'] = df['CLIENT_CREATED'].dt.strftime('%m-%d-%Y %I:%M:%S %p')
    df["CLIENT_CREATED"] = pd.to_datetime(df["CLIENT_CREATED"])

    df.dropna(inplace=True)
    df = df.sort_values("CLIENT_CREATED")
    df.set_index('CLIENT_CREATED', inplace=True)

    return df
