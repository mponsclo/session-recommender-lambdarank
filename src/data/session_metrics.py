import pandas as pd

def get_session_metrics(df: pd.DataFrame, user_id: int) -> pd.DataFrame:
    """
    Given a pandas DataFrame in the format of the train dataset and a user_id, return the following metrics for every session_id of the user:
        - user_id (int) : the given user id.
        - session_id (int) : the session id.
        - total_session_time (float) : The time passed between the first and last interactions, in seconds. Rounded to the 2nd decimal.
        - cart_addition_ratio (float) : Percentage of the added products out of the total products interacted with. Rounded ot the 2nd decimal.

    If there's no data for the given user, return an empty Dataframe preserving the expected columns.
    The column order and types must be scrictly followed.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame  of the data to be used for the agent.
    user_id : int
        Id of the client.

    Returns
    -------
    Pandas Dataframe with some metrics for all the sessions of the given user.
    """
    
    # Filter the dataframe by the given user_id
    user_id = float(user_id)
    user_df = df[df['user_id'] == user_id]
    
    # Check if the user_df is empty
    if user_df.empty:
        return pd.DataFrame(columns=['user_id', 'session_id', 'total_session_time', 'cart_addition_ratio'])
    
    # Transform user_id to int
    user_df.loc[:, 'user_id'] = user_df['user_id'].astype(int)
    
    # Calculate the total_session_time
    user_df.loc[:, 'timestamp_local'] = pd.to_datetime(user_df['timestamp_local'], errors='coerce')
    user_df = user_df.dropna(subset=['timestamp_local']).sort_values(by='timestamp_local')
    user_df['total_session_time'] = user_df.groupby('session_id')['timestamp_local'].transform(lambda x: x.max() - x.min()).round(2)
    user_df['total_session_time'] = user_df['total_session_time'].dt.total_seconds()
    
    # Calculate the cart_addition_ratio
    user_df['cart_addition_ratio'] = user_df.groupby('session_id')['add_to_cart'].transform(lambda x: x.sum() / x.count() * 100).round(2)
    user_df['cart_addition_ratio'] = user_df['cart_addition_ratio']
    # Get the unique session_id
    user_df = user_df[['user_id', 'session_id', 'total_session_time', 'cart_addition_ratio']].drop_duplicates()
    
    return user_df.sort_values(by='session_id').reset_index(drop=True)