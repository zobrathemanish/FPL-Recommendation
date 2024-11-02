import pandas as pd
import requests
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def get_player_data():
    players_url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    response = requests.get(players_url)
    if response.status_code == 200:
        data = response.json()
        players_df = pd.DataFrame(data['elements'])
        return players_df
    else:
        print(f"Failed to retrieve players data: {response.status_code} - {response.text}")
        return None

# Function to fetch fixtures data
def get_fixtures():
    fixtures_url = 'https://fantasy.premierleague.com/api/fixtures/'
    response = requests.get(fixtures_url)
    
    if response.status_code == 200:
        fixtures = pd.DataFrame(response.json())  # Fixture data
        return fixtures
    else:
        print(f"Failed to retrieve fixtures: {response.status_code}")
        return None


def get_current_gameweek():
    # FPL general information endpoint
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    
    # Make the request to FPL API
    response = requests.get(url)
    
    # If the request is successful
    if response.status_code == 200:
        data = response.json()
        
        # Get the list of events (gameweeks)
        gameweeks = data['events']
        
        # Find the current gameweek
        current_gameweek = next((gw for gw in gameweeks if gw['is_current']), None)
        
        if current_gameweek:
            return current_gameweek['id']
        else:
            return "No current gameweek found."
    else:
        return f"Error: Unable to fetch data (status code {response.status_code})"

def get_picks_for_gameweek(team_id, gw_number):
        # Construct the API URL
        url = f'https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw_number}/picks/'
        
        # Make the request to the FPL API
        response = requests.get(url)
        
        if response.status_code == 200:
            picks_data = response.json()
            picks = pd.DataFrame(picks_data['picks'])  # Picks data
            return picks
        else:
            print(f"Failed to retrieve picks: {response.status_code}")
            return None
        
def get_picks_for_gameweek(team_id, gw_number):
        # Construct the API URL
        url = f'https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw_number}/picks/'
        
        # Make the request to the FPL API
        response = requests.get(url)
        
        if response.status_code == 200:
            picks_data = response.json()
            picks = pd.DataFrame(picks_data['picks'])  # Picks data
            return picks
        else:
            print(f"Failed to retrieve picks: {response.status_code}")
            return None

def calculate_average_fdr():
        fixtures_df = get_fixtures()
        # Filter remaining (unfinished) fixtures
        remaining_fixtures = fixtures_df[fixtures_df['finished'] == False]
    
        # Create an empty list to hold FDR values for each team
        fdr_values = []
    
        # Sort the remaining fixtures by date or round to ensure the 5 upcoming are considered
        remaining_fixtures = remaining_fixtures.sort_values(by='kickoff_time')
    
        # Iterate over each team
        for team in set(remaining_fixtures['team_h']).union(set(remaining_fixtures['team_a'])):
            # Get home and away fixtures for this team
            team_fixtures = remaining_fixtures[
                (remaining_fixtures['team_h'] == team) | (remaining_fixtures['team_a'] == team)
            ].head(3)  # Limit to 1 fixtures
    
            # Collect FDR for home and away games
            for idx, fixture in team_fixtures.iterrows():
                if fixture['team_h'] == team:
                    fdr_values.append({'team': team, 'fdr': fixture['team_h_difficulty']})
                elif fixture['team_a'] == team:
                    fdr_values.append({'team': team, 'fdr': fixture['team_a_difficulty']})
    
        # Create a DataFrame from FDR values and calculate average FDR
        fdr_df = pd.DataFrame(fdr_values)
        average_fdr = fdr_df.groupby('team')['fdr'].mean().reset_index()
    
        # Rename columns
        average_fdr.columns = ['team', 'average_fdr']
    
        return average_fdr


def add_average_fdr(players_df):
    # Get average FDR for each team
    average_fdr = calculate_average_fdr()

    # Map team names back to their IDs
    team_mapping = {
        1: 'Arsenal', 2: 'Aston Villa', 3: 'Bournemouth', 4: 'Brentford',
        5: 'Brighton', 6: 'Chelsea', 7: 'Crystal Palace', 8: 'Everton',
        9: 'Fulham', 10: 'Ipswich', 11: 'Leicester', 12: 'Liverpool',
        13: 'Man City', 14: 'Man Utd', 15: 'Newcastle', 16: 'Nottingham Forest',
        17: 'Southampton', 18: 'Spurs', 19: 'West Ham', 20: 'Wolves'
    }
    
    # Map team names
    average_fdr['team_name'] = average_fdr['team'].map(team_mapping)
    
    # Invert the average FDR
    max_fdr = average_fdr['average_fdr'].max()
    average_fdr['adjusted_fdr'] = 7 - average_fdr['average_fdr']
    
    # Sort teams by adjusted FDR
    # sorted_fdr = average_fdr.sort_values(by='adjusted_fdr', ascending=False)

    # Merge adjusted FDR back to players_df
    # players_df = players_df.merge(average_fdr[['team', 'adjusted_fdr']], how='left', on='team', suffixes=('', '_avg'))
    players_df = players_df.merge(average_fdr[['team', 'adjusted_fdr','average_fdr']], how='left', on='team', suffixes=('', '_avg'))
    # players_df = players_df.merge(average_fdr, how='left', left_on='team', right_on='team')

    # Check for and drop any duplicates
    players_df = players_df.loc[:, ~players_df.columns.duplicated()]
    return players_df


def data_preprocessing():
    #Check for missing values
    players_df = get_player_data()
    fixtures_df = get_fixtures()
    # Assuming players_df and fixtures_df are already defined
    missing_values_players = players_df.isnull().sum()
    missing_values_fixtures = fixtures_df.isnull().sum()

    missing_values_summary = {
        'players_df_missing_values': missing_values_players[missing_values_players > 0],
        'fixtures_df_missing_values': missing_values_fixtures[missing_values_fixtures > 0]
    }

    print(missing_values_summary)
    #Handling missing values
        # Dropping irrelevant columns from players_df
    irrelevant_columns_players = [
        'squad_number',
        'region',
        'corners_and_indirect_freekicks_order',
        'chance_of_playing_this_round',
        'news_added',
        'chance_of_playing_next_round',
        'code', 'first_name', 'second_name', 'photo', 'news','direct_freekicks_text','penalties_text','transfers_in','transfers_out'
    ]

    # These columns are not important because we have other columns in the dataset that tells the same thing. 

    players_df_cleaned = players_df.drop(columns=irrelevant_columns_players, errors='ignore')
    # Replace NaN values in 'direct_freekick_order' and 'penalties_order' with 0
    players_df_cleaned['direct_freekicks_order'] = players_df_cleaned['direct_freekicks_order'].fillna(0)
    players_df_cleaned['penalties_order'] = players_df_cleaned['penalties_order'].fillna(0)

    # Step 2: Label encoding for the 'status' column
    if 'status' in players_df.columns:
        le = LabelEncoder()
        players_df_cleaned['status'] = le.fit_transform(players_df_cleaned['status'])

    # Convert boolean columns to integers (True -> 1, False -> 0)
    bool_columns = players_df_cleaned.select_dtypes(include=['bool']).columns
    players_df_cleaned[bool_columns] = players_df_cleaned[bool_columns].astype(int)

    # Convert object columns to numeric (if applicable)
    # This will convert the strings to NaN if they can't be converted to numbers
    object_columns = players_df_cleaned.select_dtypes(include=['object']).columns
    players_df_cleaned[object_columns] = players_df_cleaned[object_columns].apply(pd.to_numeric, errors='coerce')

    # Assuming the original status values are 'i' for injured, 'u' for unavailable, and 'a' for available
    players_df = players_df[~players_df['status'].isin(['i', 'u'])]

    # Select only the numerical columns
    data = players_df_cleaned.select_dtypes(include=['float64', 'int64']) 

    return players_df, data

def feature_selection(data):
      # Calculate the variance for each feature
    variances = data.var()

    # Sort the features by variance in descending order
    sorted_variances = variances.sort_values(ascending=False)

    # Apply VarianceThreshold to filter features with low variance
    # Set the variance threshold (e.g., 3000); you can adjust this value as needed
    selector = VarianceThreshold(threshold=0)
    selector.fit(data)

    # Get the feature names that are kept after applying VarianceThreshold
    high_variance_columns = data.columns[selector.get_support()]

    # Filter out 'id' or any other irrelevant features
    high_variance_columns = [col for col in high_variance_columns if col != 'id']

    # Print the selected high-variance features
    print("Selected high-variance features:\n", high_variance_columns)

    # Selecting features with higher variance
    data = data[high_variance_columns]

    # Sort the high variance columns based on their variance values in descending order
    high_variance_sorted = sorted_variances[high_variance_columns].sort_values(ascending=False)

    # Print the sorted high-variance features
    print("\nSorted High-Variance Features:\n", high_variance_sorted)

    #Selecting features with higher variance
    data = data[high_variance_sorted.index]
    return data

def data_scaling(data):
        # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Scale the selected features
    data_scaled = scaler.fit_transform(data)

    # Convert the scaled data back to a DataFrame
    data_scaled_df = pd.DataFrame(data_scaled, columns=data.columns)

    return data_scaled_df

def pca_and_kmeans(data_scaled_df,players_df):

    # Apply PCA
    pca = PCA(n_components=2)  # You can adjust the number of components
    data_pca = pca.fit_transform(data_scaled_df)
    explained_variance = pca.explained_variance_ratio_
    print(explained_variance)  # Output will show the proportion of variance explained by each principal component

    # Store the explained variance ratios into two variables
    explained_variance_pc1 = round(pca.explained_variance_ratio_[0],2)  # Variance for PC1
    explained_variance_pc2 = round(pca.explained_variance_ratio_[1],2)  # Variance for PC2
  
    #Check for non-linearity to assign pca score
    # Assume total_points is a Series containing the total points for each player
    total_points = players_df['total_points']  # Replace with your actual column name

    # Create a new DataFrame
    pca_df = pd.DataFrame(data=data_pca, columns=['PC1', 'PC2'])
    pca_df['Total Points'] = total_points

    pca_df_cleaned = pca_df.dropna(subset=['PC1', 'Total Points'])

    X = pca_df_cleaned[['PC1']]
    y = pca_df_cleaned['Total Points']

    # Step 2: Transform PC1 to polynomial features
    poly = PolynomialFeatures(degree=2)  # You can change the degree as needed
    X_poly = poly.fit_transform(X)

    # Step 3: Fit a polynomial regression model
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)

    # Get predictions and calculate R² score
    poly_predictions = poly_model.predict(X_poly)
    poly_r2 = r2_score(y, poly_predictions)

    # Fit a linear regression model for comparison
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    linear_predictions = linear_model.predict(X)
    linear_r2 = r2_score(y, linear_predictions)

    # Step 4: Determine non-linearity based on R² scores
    # If polynomial R² is significantly higher than linear R², we consider it non-linear
    threshold = 0.01  # Adjust threshold based on your criteria
    non_linearity = 1 if (poly_r2 - linear_r2 > threshold) else 0

    # Print results
    print(f'Polynomial R²: {poly_r2}')
    print(f'Linear R²: {linear_r2}')
    print(f'Non-linearity for PC1: {non_linearity}')

    #Handle Non-linear relationship:

    non_linearity_pc1 = 1 if poly_r2 > linear_r2 else 0

    # Step 4: Calculate Scores based on non-linearity
    if non_linearity_pc1 == 1:
        # PC1 is non-linear; use polynomial features
        poly = PolynomialFeatures(degree=2)  # Adjust degree as needed
        poly_features = poly.fit_transform(pca_df[['PC1']])  # This will include the constant term

        # Add polynomial features to the DataFrame
        pca_df[['PC1', 'PC1^2']] = poly_features[:, 1:]  # Ignore the first column (constant)

        # Calculate Score using polynomial feature of PC1 and original PC2
        players_df['Score'] = (explained_variance_pc1 * pca_df['PC1^2']) + (explained_variance_pc2 * pca_df['PC2'])
    else:
        # PC1 is linear; use original PC1
        players_df['Score'] = (explained_variance_pc1 * pca_df['PC1']) + (explained_variance_pc2 * pca_df['PC2'])

            # Step 5: Sort players_df by Score in descending order and print 'web_name' and 'Score'
        sorted_players_df = players_df[['web_name', 'Score']].sort_values(by='Score', ascending=False)

            #K means
            # Step 3: Fit K-means with the optimal number of clusters (k=2)
    optimal_k = 2
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)  # Using random_state for reproducibility
    if non_linearity_pc1:
        kmeans.fit(pca_df[['PC1', 'PC2']])  # Fit the model
    else:
        kmeans.fit(pca_df[['PC1','PC1^2', 'PC2']])  # Fit the model


    # Step 4: Add cluster labels to your original DataFrame
    pca_df['Cluster'] = kmeans.labels_

    # Optional: Print the centroids
    centroids = kmeans.cluster_centers_
    print("Cluster Centroids:")
    print(centroids)

        # Step 2: Calculate the average 'PC1' value for each cluster (you can use 'Total Points' or other metric)
    cluster_avg = pca_df.groupby('Cluster')['PC1'].mean()

    # Step 3: Identify the worst cluster (lowest average 'PC1' score)
    worst_cluster = cluster_avg.idxmin()  # Get the cluster with the lowest average 'PC1'

    # Use .loc to filter players_df based on the clusters in pca_df
    players_df_filtered = players_df.loc[pca_df['Cluster'] != worst_cluster]

    # Sort the filtered players_df by Score in descending order
    players_df_filtered_sorted = players_df_filtered.sort_values(by='Score', ascending=False)

    # Display the sorted DataFrame
    print(players_df_filtered_sorted[['web_name', 'Score']])

    players_df_filtered = add_average_fdr(players_df_filtered)

    return players_df, players_df_filtered

def adjust_fdr(players_df_filtered):
    
    # Calculate the new score
    # Set the weight for adjusted_fdr
    weight = 0.1  # Adjust this value based on how much you want to influence the score

    # Calculate the new score
    players_df_filtered['adjusted_score'] = players_df_filtered['Score'] + (weight * players_df_filtered['adjusted_fdr'])

    # Optional: Display the updated DataFrame to check the new scores
    print(players_df_filtered[['Score', 'adjusted_fdr', 'adjusted_score']].head())

    # Sort the filtered players_df by Score in descending order
    players_df_filtered_sorted = players_df_filtered.sort_values(by='Score', ascending=False)

    # Display the sorted DataFrame
    print(players_df_filtered_sorted[['web_name', 'Score']])

    # Continue to map positions and get top players

    # Position mapping
    position_mapping = {1: 'goalkeeper', 2: 'defender', 3: 'midfielder', 4: 'forward'}
    players_df_filtered_sorted['position'] = players_df_filtered_sorted['element_type'].map(position_mapping)

    players_df_filtered_sorted = add_average_fdr(players_df_filtered_sorted)
    return players_df_filtered_sorted
    


def get_top_players_by_position():
    # Function to fetch fixtures data
    players_df, pre_data = data_preprocessing()
    data = feature_selection(pre_data)
    data_scaled_df = data_scaling(data)
    players_df, players_df_filtered  = pca_and_kmeans(data_scaled_df,players_df)
    players_df_filtered_sorted = adjust_fdr(players_df_filtered)
    # Rank players based on PCA score from the filtered dataframe
    top_goalkeepers = players_df_filtered_sorted[players_df_filtered_sorted['position'] == 'goalkeeper']\
                        .nlargest(15, 'adjusted_score')[['id', 'web_name', 'adjusted_score', 'now_cost', 'influence', 'average_fdr', 'minutes', 'element_type']]

    top_defenders = players_df_filtered_sorted[players_df_filtered_sorted['position'] == 'defender']\
                        .nlargest(15, 'adjusted_score')[['id', 'web_name', 'adjusted_score', 'now_cost', 'influence', 'average_fdr', 'minutes','element_type' ]]

    top_midfielders = players_df_filtered_sorted[players_df_filtered_sorted['position'] == 'midfielder']\
                        .nlargest(15, 'adjusted_score')[['id', 'web_name', 'adjusted_score', 'now_cost', 'influence', 'average_fdr', 'minutes', 'element_type']]

    top_forwards = players_df_filtered_sorted[players_df_filtered_sorted['position'] == 'forward']\
                        .nlargest(20, 'adjusted_score')[['id', 'web_name', 'adjusted_score', 'now_cost', 'influence', 'average_fdr', 'minutes', 'element_type']]


    # # Return top players
    # return top_goalkeepers, top_defenders, top_midfielders, top_forwards

    # Display results
    print("Top Goalkeepers:")
    print(top_goalkeepers)

    print("\nTop Defenders:")
    print(top_defenders)

    print("\nTop Midfielders:")
    print(top_midfielders)

    print("\nTop Forwards:")
    print(top_forwards)

    return top_goalkeepers,top_defenders,top_midfielders,top_forwards, players_df_filtered_sorted

def get_input_squad(myteam_id, gw_number):
    # Function to fetch the picks for a given team and gameweek
    # Fetch the player data and picks for a specific gameweek
    players_df = get_player_data()  # Unpack the tuple here
    picks_df = get_picks_for_gameweek(myteam_id, gw_number)

    # Extract important player details like name and ID from the player data
    player_details = players_df[['id', 'web_name']]
    # Check if either DataFrame is None
    if picks_df is None:
        print("Error: picks_df is None")
    if player_details is None:
        print("Error: player_details is None")
    
    # Merge the picks with player details using the 'element' (player ID)
    merged_df = pd.merge(picks_df, player_details, left_on='element', right_on='id', how='left')
    
    # Create the list of player names and their IDs
    input_squad = merged_df[['id', 'web_name']]
    
    return input_squad


def get_squad_stats(input_squad, player_df):
    # Filter player_df for players in the input squad
    squad_stats = player_df[player_df['id'].isin(input_squad['id'])]

    # Select relevant columns to display
    relevant_columns = ['id','ep_next', 'web_name', 'element_type', 'now_cost', 
                        'minutes', 'goals_scored', 'assists', 
                        'influence', 'points_per_game','points_per_game','selected_rank']
    
    # Create a new DataFrame with the relevant stats
    stats_to_display = squad_stats[relevant_columns]
    
   # Sort the DataFrame by 'now_cost'
    stats_to_display = stats_to_display.sort_values(by='now_cost', ascending=True)
    
def select_starting_11_and_bench(input_squad, players_df):
    """
    This function merges the input_squad with players_df to retrieve the required player details
    and then splits the squad into starting 11 and bench according to constraints.
    
    Bench constraints:
    - 1 goalkeeper at most
    - Maximum of 2 defenders
    - Maximum of 2 forwards
    - 4 players in total

    The lowest-cost players will be selected for the bench following these constraints.
    
    Parameters:
    input_squad (DataFrame): DataFrame containing 'id' and 'web_name' of selected players.
    players_df (DataFrame): DataFrame containing full player information, including 'id', 'element_type', 'now_cost'.
    
    Returns:
    tuple: (starting_11, bench) both as lists of player dictionaries.
    """
    players_df = get_player_data()
    # Merge input_squad with players_df on 'id' or 'web_name'
    merged_squad = pd.merge(input_squad, players_df, on='id', how='left')

    # Convert the merged DataFrame to a list of dictionaries for easier processing
    squad = merged_squad.to_dict(orient='records')

    # Separate players by position
    goalkeepers = [player for player in squad if player['element_type'] == 1]
    defenders = [player for player in squad if player['element_type'] == 2]
    midfielders = [player for player in squad if player['element_type'] == 3]
    forwards = [player for player in squad if player['element_type'] == 4]

    # Sort all players by 'now_cost' in ascending order
    sorted_squad = sorted(squad, key=lambda x: x['now_cost'])

    bench = []

    # Add 1 goalkeeper to the bench if available
    for player in sorted_squad:
        if len(bench) < 4 and player['element_type'] == 1 and sum(1 for p in bench if p['element_type'] == 1) < 1:
            bench.append(player)

    # Add up to 2 defenders to the bench
    for player in sorted_squad:
        if len(bench) < 4 and player['element_type'] == 2 and sum(1 for p in bench if p['element_type'] == 2) < 2:
            bench.append(player)

    # Add up to 2 forwards to the bench
    for player in sorted_squad:
        if len(bench) < 4 and player['element_type'] == 4 and sum(1 for p in bench if p['element_type'] == 4) < 2:
            bench.append(player)

    # If bench isn't full yet, fill remaining slots with midfielders
    for player in sorted_squad:
        if len(bench) < 4 and player['element_type'] == 3:
            bench.append(player)

    # Ensure the bench is exactly 4 players (if less, fill remaining with the next lowest cost players)
    while len(bench) < 4:
        for player in sorted_squad:
            if player not in bench:
                bench.append(player)
                break

    # Remaining players form the starting 11
    starting_11 = [player for player in squad if player not in bench]

    return starting_11, bench

def assign_bench_and_watch_avoid(input_squad):
     # Fetch the player data and picks for a specific gameweek
    players_df = get_player_data()  # Unpack the tuple here
    
    # Simulate top players by position (replace with your real data fetching function)
    top_goalkeepers, top_defenders, top_midfielders, top_forwards, _ = get_top_players_by_position()

    # Add a 'position' column to each DataFrame
    top_goalkeepers['position'] = 'Goalkeeper'
    top_defenders['position'] = 'Defender'
    top_midfielders['position'] = 'Midfielder'
    top_forwards['position'] = 'Forward'

    # Concatenate all players into a single DataFrame
    all_top_players = pd.concat([top_goalkeepers, top_defenders, top_midfielders, top_forwards], ignore_index=True)

    # Ensure columns 'influence' and 'now_cost' are numeric
    all_top_players['influence'] = pd.to_numeric(all_top_players['influence'], errors='coerce')
    all_top_players['now_cost'] = pd.to_numeric(all_top_players['now_cost'], errors='coerce')
    
    # Drop rows with NaN in 'influence' or 'now_cost'
    all_top_players.dropna(subset=['influence', 'now_cost'], inplace=True)

    # Keep players who are in both input squad and all_top_players
    keep = all_top_players[all_top_players['id'].isin(input_squad)]
    
    # Players in input_squad but not in the top players (watch/avoid)
    remaining_watch_avoid = [player for player in input_squad if player not in keep['id'].values]
    
    # Get player names for the remaining_watch_avoid list
    remaining_watch_avoid_names = players_df[players_df['id'].isin(remaining_watch_avoid)]
    
    # Return the final lists
    return keep, remaining_watch_avoid_names

def calculate_transfer_budget(team_id, gw_number):
    import pandas as pd
    # Function to fetch the picks for a given team and gameweek

    # Fetch the player data and picks for a specific gameweek
    players_df = get_player_data()  # Unpack the tuple here
    picks_df = get_picks_for_gameweek(team_id, gw_number)
    
    # Extract important player details like name and ID from the player data
    player_details = players_df[['id', 'web_name','now_cost']]
    
    # Merge the picks with player details using the 'element' (player ID)
    merged_df = pd.merge(picks_df, player_details, left_on='element', right_on='id', how='left')
    # Calculate the total transfer budget
    transfer_budget = merged_df['now_cost'].sum() / 10  # now_cost is in tenths of millions, convert to millions
    return transfer_budget
    
    import pandas as pd
import requests

def get_team_info(team_id):
    # Construct the API URL for team details
    url = f'https://fantasy.premierleague.com/api/entry/{team_id}/'
    
    # Make the request to the FPL API
    response = requests.get(url)
    
    if response.status_code == 200:
        team_data = response.json()
        # print(team_data)  # Print the entire team data for inspection
        return team_data
    else:
        print(f"Failed to retrieve team info: {response.status_code}")
        return None

def get_free_transfers(team_id):
    team_info = get_team_info(team_id)
    
    if team_info is None:
        return None

    # Extract required fields from team_info
    last_deadline_bank = team_info.get('last_deadline_bank', 0)
    last_deadline_value = team_info.get('last_deadline_value', 0)  # This may not be needed for transfers
    last_deadline_total_transfers = team_info.get('last_deadline_total_transfers', 0)

    # print (current_gameweek)
    # Calculate available transfers
    current_gameweek = team_info.get('current_event', 0)
    available_transfers = current_gameweek - last_deadline_total_transfers
    
    
    # Return the calculated values
    return {
        'last_deadline_bank': last_deadline_bank,
        'available_transfers': available_transfers,
    }

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

def sort_by_pca_scores(watch_avoid):
    # Fetch data
    fixtures_df = get_fixtures()
    players_df = get_player_data()
    
    if players_df is None or fixtures_df is None:
        print("Failed to retrieve necessary data. Exiting.")
        return None

    # Get average FDR for each team
    average_fdr = calculate_average_fdr()

    # Map team names back to their IDs
    team_mapping = {
            1: 'Arsenal', 2: 'Aston Villa', 3: 'Bournemouth', 4: 'Brentford',
            5: 'Brighton', 6: 'Chelsea', 7: 'Crystal Palace', 8: 'Everton',
            9: 'Fulham', 10: 'Ipswich', 11: 'Leicester', 12: 'Liverpool',
            13: 'Man City', 14: 'Man Utd', 15: 'Newcastle', 16: 'Nottingham Forest',
            17: 'Southampton', 18: 'Spurs', 19: 'West Ham', 20: 'Wolves'
    }
    
    average_fdr['team_name'] = average_fdr['team'].map(team_mapping)
    watch_avoid = watch_avoid.merge(average_fdr, how='left', left_on='team', right_on='team')
    
    # Define the features necessary for clustering
    feature_columns = ['ep_this', 'minutes', 'assists', 'bps', 'influence', 'creativity', 'ict_index', 'average_fdr']

    # Drop rows with missing data in the feature columns
    watch_avoid = watch_avoid.dropna(subset=feature_columns)

    # Convert features to numeric and drop any rows with invalid values
    X = watch_avoid[feature_columns].apply(pd.to_numeric, errors='coerce').dropna()

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=2)  # Use two components
    pca_scores = pca.fit_transform(X_scaled)

    # Add PCA scores to the DataFrame
    watch_avoid['adjusted_score'] = pca_scores[:, 0]  # Take the first component as the score

    # Apply K-Means clustering (using 2 clusters as an example)
    kmeans = KMeans(n_clusters=2, random_state=42)
    watch_avoid['cluster'] = kmeans.fit_predict(X_scaled)

    # Rank players based on PCA scores (higher score is better)
    watch_avoid['rank'] = watch_avoid['adjusted_score'].rank(ascending=False)

    # Sort the DataFrame by rank
    sorted_watch_avoid = watch_avoid.sort_values(by='rank')

    # Return only 'id', 'web_name', and 'rank' columns
    return sorted_watch_avoid[['id', 'web_name', 'rank', 'adjusted_score','element_type','now_cost']]

def suggest_transfer(watch_avoid, top_goalkeepers, top_defenders, top_midfielders, top_forwards, available_budget, free_transfers, team_id):
    # Define position mapping
    position_map = {1: 'Goalkeeper', 2: 'Defender', 3: 'Midfielder', 4: 'Forward'}

    # Function to find replacement players from top players
    def find_replacement(player, top_players, available_budget, team_id):
        gw_number = get_current_gameweek()
        input_squad = get_input_squad(team_id, gw_number)

        # Filter top players who are equal to or cheaper than (player's cost + available budget)
        affordable_players = top_players[top_players['now_cost'] <= (player['now_cost'] + available_budget)]

        # Further filter players to those with higher PCA scores
        affordable_players = affordable_players[affordable_players['adjusted_score'] > player['adjusted_score']]

        # Exclude players who are already in the squad
        affordable_players = affordable_players[~affordable_players['web_name'].isin(input_squad['web_name'])]

        if not affordable_players.empty:
            # Sort affordable players by PCA score (descending for higher scores)
            affordable_players = affordable_players.sort_values(by='adjusted_score', ascending=False)
            # Return the best replacement (highest PCA score)
            print("returned players are, ", affordable_players.iloc[0])
            return affordable_players.iloc[0]

        return None

    # List to store suggested transfers
    suggested_transfers = []
    print("columns are", watch_avoid.columns)
     # Position mapping
    # position_mapping = {1: 'goalkeeper', 2: 'defender', 3: 'midfielder', 4: 'forward'}
    # players_df_filtered_sorted['position'] = players_df_filtered_sorted['element_type'].map(position_mapping)

    # Evaluate transfers based on free transfers available
    for index, player in watch_avoid.iterrows():
        player_position = player['element_type']  # 1 for GK, 2 for DEF, etc.
        
        # Determine the top players based on the player's position
        if player_position == 1:
            top_players = top_goalkeepers
        elif player_position == 2:
            top_players = top_defenders
        elif player_position == 3:
            top_players = top_midfielders
        elif player_position == 4:
            top_players = top_forwards
        
        # Find a replacement player for the "avoid" player
        replacement = find_replacement(player, top_players, available_budget, team_id)
        if replacement is not None:
            suggested_transfers.append({
                'sell': player['web_name'],
                'buy': replacement['web_name'],
                'position': position_map[player_position],
                'sell_price': player['now_cost'],
                'buy_price': replacement['now_cost']
            })
        
        # Break if we have enough transfers for free transfers
        if len(suggested_transfers) == free_transfers:
            break

    # if suggested_transfers:
    #     # Print out the suggested transfers
    #     for transfer in suggested_transfers:
    #         print(f"Suggesting Transfer: Sell {transfer['sell']} ({transfer['position']}) for {transfer['sell_price'] / 10}m")
    #         print(f"Buy {transfer['buy']} ({transfer['position']}) for {transfer['buy_price'] / 10}m\n")
    # else:
    #     print("No suitable transfers found.")

    return suggested_transfers

# def suggest_multiple_transfers(watch_avoid, top_goalkeepers, top_defenders, top_midfielders, top_forwards, available_budget, free_transfers, team_id):
#     """
#     This function suggests multiple transfers by selling multiple players and buying multiple replacements based on PCA scores.

#     Parameters:
#     - watch_avoid (DataFrame): The list of players to consider for selling.
#     - top_goalkeepers, top_defenders, top_midfielders, top_forwards (DataFrame): DataFrames of top players by position.
#     - available_budget (float): The budget left for transfers.
#     - free_transfers (int): Number of free transfers available.
#     - team_id (int): The user's team ID.

#     Returns:
#     - List of transfer suggestions (sell and buy pairs).
#     """
#     position_map = {1: 'Goalkeeper', 2: 'Defender', 3: 'Midfielder', 4: 'Forward'}

#     # Function to find replacements for a combination of players to sell
#     def find_replacements(players_to_sell, available_budget, team_id):
#         gw_number = get_current_gameweek()
#         input_squad = get_input_squad(team_id, gw_number)

#         replacements = []
#         remaining_budget = available_budget  # Start with only the available budget

#         # Iterate through the players_to_sell and find replacements
#         for player in players_to_sell:
#             player_position = player['element_type']

#             # Get the corresponding top players for the player's position
#             if player_position == 1:
#                 top_players = top_goalkeepers
#             elif player_position == 2:
#                 top_players = top_defenders
#             elif player_position == 3:
#                 top_players = top_midfielders
#             elif player_position == 4:
#                 top_players = top_forwards

#             # Filter top players by available budget and higher PCA score than the player to be sold
#             affordable_players = top_players[top_players['now_cost'] <= remaining_budget]
#             affordable_players = affordable_players[affordable_players['pca_score'] > player['pca_score']]

#             # Exclude players already in the squad
#             affordable_players = affordable_players[~affordable_players['web_name'].isin(input_squad['web_name'])]

#             if not affordable_players.empty:
#                 # Sort by PCA score and select the best replacement
#                 affordable_players = affordable_players.sort_values(by='pca_score', ascending=False)
#                 best_replacement = affordable_players.iloc[0]
#                 replacements.append(best_replacement)

#                 # Update remaining budget
#                 remaining_budget -= best_replacement['now_cost']

#         return replacements if replacements else None

#     # List to store suggested transfers
#     suggested_transfers = []

#     # Make a copy of watch_avoid to avoid modifying the original DataFrame
#     remaining_watch_avoid = watch_avoid.copy()

#     # Iterate over combinations of players in watch_avoid
#     for i in range(len(remaining_watch_avoid)):
#         for j in range(i + 1, len(remaining_watch_avoid)):
#             # Get the current two players to sell
#             players_to_sell = [remaining_watch_avoid.iloc[i], remaining_watch_avoid.iloc[j]]

#             # Find replacements for the combination of players
#             replacements = find_replacements(players_to_sell, available_budget, team_id)

#             if replacements:
#                 # Calculate total selling prices
#                 total_selling_price = sum(player['now_cost'] for player in players_to_sell) + available_budget
#                 # Calculate total buying prices
#                 total_buying_price = sum(replacement['now_cost'] for replacement in replacements)

#                 # Check if total selling price + available budget is greater than or equal to total buying price
#                 if total_selling_price >= total_buying_price:
#                     for sell_player, buy_player in zip(players_to_sell, replacements):
#                         suggested_transfers.append({
#                             'sell': sell_player['web_name'],
#                             'buy': buy_player['web_name'],
#                             'position': position_map[sell_player['element_type']],
#                             'sell_price': float(sell_player['now_cost']),
#                             'buy_price': float(buy_player['now_cost'])
#                         })

#                     # Remove the processed players by their actual index, not the positional index
#                     indices_to_drop = [remaining_watch_avoid.iloc[i].name, remaining_watch_avoid.iloc[j].name]
#                     remaining_watch_avoid = remaining_watch_avoid.drop(indices_to_drop)

#                 # Break if enough transfers for free_transfers are suggested
#                 if len(suggested_transfers) >= free_transfers:
#                     break

#         if len(suggested_transfers) >= free_transfers:
#             break

#     # Output suggested transfers
#     if suggested_transfers:
#         for transfer in suggested_transfers:
#             print(f"Suggesting Transfer: Sell {transfer['sell']} ({transfer['position']}) for {transfer['sell_price'] / 10}m")
#             print(f"Buy {transfer['buy']} ({transfer['position']}) for {transfer['buy_price'] / 10}m\n")
#     else:
#         print("No suitable transfers found.")

#     return suggested_transfers


def determine_captain_and_vice_captain(players_df):
    # Ensure the necessary columns are present
    if 'adjusted_score' not in players_df.columns:
        print("PCA score column is missing in the DataFrame.")
        return None

    # Drop rows with missing PCA scores
    players_df = players_df.dropna(subset=['adjusted_score'])

    # Check if there are enough players to select a captain and vice-captain
    if players_df.shape[0] < 2:
        print("Not enough players to determine captain and vice-captain.")
        return None

    # Sort the players by PCA score in descending order
    sorted_players = players_df.sort_values(by='adjusted_score', ascending=False)

    # Select the captain and vice-captain based on PCA scores
    captain = sorted_players.iloc[0]  # Player with the highest PCA score
    vice_captain = sorted_players.iloc[1]  # Player with the second-highest PCA score

    # Return the results
    return {
        'captain': captain[['web_name']],
        'vice_captain': vice_captain[['web_name']]
    }
