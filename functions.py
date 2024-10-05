import pandas as pd
import requests
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

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

def calculate_average_fdr(fixtures_df):
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
            ].head(5)  # Limit to 5 fixtures
    
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

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder

def get_intracluster_distance(player_name):
    top_goalkeepers, top_defenders, top_midfielders, top_forwards, all_players = get_top_players_by_position()
    
    # Drop rows with NaN in the 'web_name' column before performing operations
    all_players = all_players[all_players['web_name'].notna()]

    # Convert web_name to lowercase for case-insensitive comparison
    all_players['web_name'] = all_players['web_name'].str.lower()

    # Search for the player by name (also lowercased)
    player_row = all_players[all_players['web_name'] == player_name.lower()]

    if not player_row.empty:
        # If the player exists, return their PCA-based ranking score
        player_pca_score = player_row['pca_score'].values[0]
        return player_pca_score
    else:
        print(f"Player '{player_name}' not found.")
        return None

def get_top_players_by_position():
    # Function to fetch fixtures data
    def get_fixtures():
        fixtures_url = 'https://fantasy.premierleague.com/api/fixtures/'
        response = requests.get(fixtures_url)
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        else:
            print(f"Failed to retrieve fixtures: {response.status_code} - {response.text}")
            return None

    # Fetch data
    fixtures_df = get_fixtures()
    players_df = get_player_data()

    if players_df is None or fixtures_df is None:
        print("Failed to retrieve necessary data. Exiting.")
        return None

    # Filter out players with status 'i' (injured) or 'u' (unavailable)
    players_df = players_df[~players_df['status'].isin(['i', 'u'])]

    # Get average FDR for each team
    average_fdr = calculate_average_fdr(fixtures_df)

    # Map team names back to their IDs
    team_mapping = {
        1: 'Arsenal', 2: 'Aston Villa', 3: 'Bournemouth', 4: 'Brentford',
        5: 'Brighton', 6: 'Chelsea', 7: 'Crystal Palace', 8: 'Everton',
        9: 'Fulham', 10: 'Ipswich', 11: 'Leicester', 12: 'Liverpool',
        13: 'Man City', 14: 'Man Utd', 15: 'Newcastle', 16: 'Nottingham Forest',
        17: 'Southampton', 18: 'Spurs', 19: 'West Ham', 20: 'Wolves'
    }
    
    average_fdr['team_name'] = average_fdr['team'].map(team_mapping)
    players_df = players_df.merge(average_fdr, how='left', left_on='team', right_on='team')

    # Label encoding for the 'status' column
    if 'status' in players_df.columns:
        le = LabelEncoder()
        players_df['status'] = le.fit_transform(players_df['status'])

    # Define the feature columns for PCA
    feature_columns = ['ep_this', 'minutes', 'goals_scored', 'assists', 'bps', 'influence', 'creativity', 'ict_index', 'average_fdr']
    
    # Preprocessing
    players_df = players_df.dropna(subset=feature_columns)
    X = players_df[feature_columns].apply(pd.to_numeric, errors='coerce').dropna()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=2)  # Reduce to 2 components
    X_pca = pca.fit_transform(X_scaled)

    # Store the PCA results in the dataframe
    players_df['pca_1'] = X_pca[:, 0]
    players_df['pca_2'] = X_pca[:, 1]

    # Calculate the PCA score (sum of the two components)
    players_df['pca_score'] = players_df['pca_1'] + players_df['pca_2']

    # Apply K-Means Clustering based on PCA results
    kmeans = KMeans(n_clusters=2, random_state=42)
    players_df['cluster'] = kmeans.fit_predict(X_pca)

    # Rank the players based on the PCA score (higher score is better)
    players_df['rank'] = players_df['pca_score'].rank(ascending=False)

    # Sort the DataFrame by rank
    players_df_sorted = players_df.sort_values(by='rank')

    # Calculate the mean PCA score for each cluster
    cluster_means = players_df.groupby('cluster')['pca_score'].mean().reset_index()

    # Identify the best cluster based on the higher mean PCA score
    best_cluster = cluster_means.loc[cluster_means['pca_score'].idxmax(), 'cluster']
    
    # Filter to only include players from the best cluster
    players_df_filtered = players_df_sorted[players_df_sorted['cluster'] == best_cluster]
   
    # Position mapping
    position_mapping = {1: 'goalkeeper', 2: 'defender', 3: 'midfielder', 4: 'forward'}
    players_df_filtered['position'] = players_df_filtered['element_type'].map(position_mapping)

    # Rank players based on PCA score from the filtered dataframe
    top_goalkeepers = players_df_filtered[players_df_filtered['position'] == 'goalkeeper'].nlargest(5, 'pca_score')[['id', 'web_name', 'pca_score', 'now_cost', 'influence', 'average_fdr', 'minutes', 'rank', 'element_type']]
    top_defenders = players_df_filtered[players_df_filtered['position'] == 'defender'].nlargest(15, 'pca_score')[['id', 'web_name', 'pca_score', 'now_cost', 'influence', 'average_fdr', 'minutes', 'rank', 'element_type']]
    top_midfielders = players_df_filtered[players_df_filtered['position'] == 'midfielder'].nlargest(15, 'pca_score')[['id', 'web_name', 'pca_score', 'now_cost', 'influence', 'average_fdr', 'minutes', 'rank', 'element_type']]
    top_forwards = players_df_filtered[players_df_filtered['position'] == 'forward'].nlargest(13, 'pca_score')[['id', 'web_name', 'pca_score', 'now_cost', 'influence', 'average_fdr', 'minutes', 'rank', 'element_type']]

    return top_goalkeepers, top_defenders, top_midfielders, top_forwards, players_df_filtered

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
    average_fdr = calculate_average_fdr(fixtures_df)

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
    watch_avoid['pca_score'] = pca_scores[:, 0]  # Take the first component as the score

    # Apply K-Means clustering (using 2 clusters as an example)
    kmeans = KMeans(n_clusters=2, random_state=42)
    watch_avoid['cluster'] = kmeans.fit_predict(X_scaled)

    # Rank players based on PCA scores (higher score is better)
    watch_avoid['rank'] = watch_avoid['pca_score'].rank(ascending=False)

    # Sort the DataFrame by rank
    sorted_watch_avoid = watch_avoid.sort_values(by='rank')

    # Return only 'id', 'web_name', and 'rank' columns
    return sorted_watch_avoid[['id', 'web_name', 'rank', 'pca_score','element_type','now_cost']]

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
        affordable_players = affordable_players[affordable_players['pca_score'] > player['pca_score']]

        # Exclude players who are already in the squad
        affordable_players = affordable_players[~affordable_players['web_name'].isin(input_squad['web_name'])]

        if not affordable_players.empty:
            # Sort affordable players by PCA score (descending for higher scores)
            affordable_players = affordable_players.sort_values(by='pca_score', ascending=False)
            # Return the best replacement (highest PCA score)
            return affordable_players.iloc[0]

        return None

    # List to store suggested transfers
    suggested_transfers = []

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

def suggest_multiple_transfers(watch_avoid, top_goalkeepers, top_defenders, top_midfielders, top_forwards, available_budget, free_transfers, team_id):
    """
    This function suggests multiple transfers by selling multiple players and buying multiple replacements based on PCA scores.

    Parameters:
    - watch_avoid (DataFrame): The list of players to consider for selling.
    - top_goalkeepers, top_defenders, top_midfielders, top_forwards (DataFrame): DataFrames of top players by position.
    - available_budget (float): The budget left for transfers.
    - free_transfers (int): Number of free transfers available.
    - team_id (int): The user's team ID.

    Returns:
    - List of transfer suggestions (sell and buy pairs).
    """
    position_map = {1: 'Goalkeeper', 2: 'Defender', 3: 'Midfielder', 4: 'Forward'}

    # Function to find replacements for a combination of players to sell
    def find_replacements(players_to_sell, available_budget, team_id):
        gw_number = get_current_gameweek()
        input_squad = get_input_squad(team_id, gw_number)

        # Get total selling price for selected players
        total_selling_price = sum(player['now_cost'] for player in players_to_sell)

        # Combine available budget with total selling price to get total budget for buying new players
        combined_budget = total_selling_price + available_budget

        replacements = []
        remaining_budget = combined_budget

        # Iterate through the players_to_sell and find replacements
        for player in players_to_sell:
            player_position = player['element_type']

            # Get the corresponding top players for the player's position
            if player_position == 1:
                top_players = top_goalkeepers
            elif player_position == 2:
                top_players = top_defenders
            elif player_position == 3:
                top_players = top_midfielders
            elif player_position == 4:
                top_players = top_forwards

            # Filter top players by available budget and higher PCA score than the player to be sold
            affordable_players = top_players[top_players['now_cost'] <= remaining_budget]
            affordable_players = affordable_players[affordable_players['pca_score'] > player['pca_score']]

            # Exclude players already in the squad
            affordable_players = affordable_players[~affordable_players['web_name'].isin(input_squad['web_name'])]

            if not affordable_players.empty:
                # Sort by PCA score and select the best replacement
                affordable_players = affordable_players.sort_values(by='pca_score', ascending=False)
                best_replacement = affordable_players.iloc[0]
                replacements.append(best_replacement)

                # Update remaining budget
                remaining_budget -= best_replacement['now_cost']

        return replacements if replacements else None

    # List to store suggested transfers
    suggested_transfers = []

    # Make a copy of watch_avoid to avoid modifying the original DataFrame
    remaining_watch_avoid = watch_avoid.copy()

    # Iterate over combinations of players in watch_avoid, starting from the bottom 2
    for i in range(len(remaining_watch_avoid)):
        for j in range(i + 1, len(remaining_watch_avoid)):
            # Get the current two players to sell
            players_to_sell = [remaining_watch_avoid.iloc[i], remaining_watch_avoid.iloc[j]]

            # Find replacements for the combination of players
            replacements = find_replacements(players_to_sell, available_budget, team_id)

            if replacements:
                # Add each replacement suggestion to the suggested_transfers list
                for sell_player, buy_player in zip(players_to_sell, replacements):
                    suggested_transfers.append({
                        'sell': sell_player['web_name'],
                        'buy': buy_player['web_name'],
                        'position': position_map[sell_player['element_type']],
                        'sell_price': int(sell_player['now_cost']/10),
                        'buy_price': int(buy_player['now_cost']/10)
                    })

                # Remove the processed players by their actual index, not the positional index
                indices_to_drop = [remaining_watch_avoid.iloc[i].name, remaining_watch_avoid.iloc[j].name]
                remaining_watch_avoid = remaining_watch_avoid.drop(indices_to_drop)

            # Break if enough transfers for free_transfers are suggested
            if len(suggested_transfers) >= free_transfers:
                break

        if len(suggested_transfers) >= free_transfers:
            break

    # Output suggested transfers
    if suggested_transfers:
        for transfer in suggested_transfers:
            print(f"Suggesting Transfer: Sell {transfer['sell']} ({transfer['position']}) for {transfer['sell_price'] / 10}m")
            print(f"Buy {transfer['buy']} ({transfer['position']}) for {transfer['buy_price'] / 10}m\n")
    else:
        print("No suitable transfers found.")

    return suggested_transfers



def determine_captain_and_vice_captain(players_df):
    # Ensure the necessary columns are present
    if 'pca_score' not in players_df.columns:
        print("PCA score column is missing in the DataFrame.")
        return None

    # Drop rows with missing PCA scores
    players_df = players_df.dropna(subset=['pca_score'])

    # Check if there are enough players to select a captain and vice-captain
    if players_df.shape[0] < 2:
        print("Not enough players to determine captain and vice-captain.")
        return None

    # Sort the players by PCA score in descending order
    sorted_players = players_df.sort_values(by='pca_score', ascending=False)

    # Select the captain and vice-captain based on PCA scores
    captain = sorted_players.iloc[0]  # Player with the highest PCA score
    vice_captain = sorted_players.iloc[1]  # Player with the second-highest PCA score

    # Return the results
    return {
        'captain': captain[['web_name']],
        'vice_captain': vice_captain[['web_name']]
    }
