import pandas as pd
import requests
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

def hello():
    print("hello you are here")
    
def get_players_data():
    players_url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    response = requests.get(players_url)
    if response.status_code == 200:
        data = response.json()
        players_df = pd.DataFrame(data['elements'])
        return players_df
    else:
        print(f"Failed to retrieve players data: {response.status_code} - {response.text}")
        return None

def get_intracluster_distance(player_name):
    top_goalkeepers, top_defenders, top_midfielders, top_forwards, all_players = get_top_players_by_position()
    
    # Drop rows with NaN in the 'web_name' column before performing operations
    all_players = all_players[all_players['web_name'].notna()]

    # Convert web_name to lowercase for case-insensitive comparison
    all_players['web_name'] = all_players['web_name'].str.lower()

    # Search for the player by name (also lowercased)
    player_row = all_players[all_players['web_name'] == player_name.lower()]

    if not player_row.empty:
        # If the player exists, return their intracluster distance
        player_intracluster_distance = player_row['intracluster_distance'].values[0]
        return player_intracluster_distance
    else:
        print(f"Player '{player_name}' not found.")
        return None  # or return a suitable default value


    
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
        


    def calculate_average_fdr(fixtures_df):
        remaining_fixtures = fixtures_df[fixtures_df['finished'] == False]
        fdr_values = []
        for idx, fixture in remaining_fixtures.iterrows():
            team_h = fixture['team_h']
            team_a = fixture['team_a']
            fdr_h = fixture['team_h_difficulty']
            fdr_a = fixture['team_a_difficulty']
            fdr_values.append({'team': team_h, 'fdr': fdr_h})
            fdr_values.append({'team': team_a, 'fdr': fdr_a})
        fdr_df = pd.DataFrame(fdr_values)
        average_fdr = fdr_df.groupby('team')['fdr'].mean().reset_index()
        average_fdr.columns = ['team', 'average_fdr']
        return average_fdr

    # Fetch data
    fixtures_df = get_fixtures()
    players_df = get_players_data()
    
    
    if players_df is None or fixtures_df is None:
        print("Failed to retrieve necessary data. Exiting.")
        return None

    # Get average FDR for each team
    average_fdr = calculate_average_fdr(fixtures_df)

    # Map team names back to their IDs
    team_mapping = {
        1: 'Arsenal', 2: 'Aston Villa', 3: 'Brentford', 4: 'Brighton',
        # Add the remaining teams here...
    }
    average_fdr['team_name'] = average_fdr['team'].map(team_mapping)
    players_df = players_df.merge(average_fdr, how='left', left_on='team', right_on='team')

    # Label encoding for the 'status' column
    if 'status' in players_df.columns:
        le = LabelEncoder()
        players_df['status'] = le.fit_transform(players_df['status'])

    # Define the feature columns for clustering
    feature_columns = ['ep_this', 'minutes', 'assists', 'bps', 'influence', 'creativity', 'ict_index', 'average_fdr']
    
    # Preprocessing
    players_df = players_df.dropna(subset=feature_columns)
    X = players_df[feature_columns].apply(pd.to_numeric, errors='coerce').dropna()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply K-Means Clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    players_df['cluster'] = kmeans.fit_predict(X_scaled)

    # Calculate intracluster distances
    def calculate_intracluster_distance(cluster_data):
        distances = np.linalg.norm(cluster_data[:, np.newaxis] - cluster_data, axis=2)
        avg_distances = distances.mean(axis=1)
        return avg_distances

    players_df['intracluster_distance'] = np.nan
    for cluster in players_df['cluster'].unique():
        cluster_data = X_scaled[players_df['cluster'] == cluster]
        intracluster_distances = calculate_intracluster_distance(cluster_data)
        players_df.loc[players_df['cluster'] == cluster, 'intracluster_distance'] = intracluster_distances

    position_mapping = {1: 'goalkeeper', 2: 'defender', 3: 'midfielder', 4: 'forward'}
    players_df['position'] = players_df['element_type'].map(position_mapping)

    # Determine the best cluster based on average intracluster distance
    avg_intracluster_distances = players_df.groupby('cluster')['intracluster_distance'].mean()
    best_cluster = avg_intracluster_distances.idxmax()  # Cluster with the lowest average distance

    # Rank players based on the best cluster
    top_potential_players = players_df[players_df['cluster'] == best_cluster]

    top_goalkeepers = top_potential_players[top_potential_players['position'] == 'goalkeeper'].nlargest(15, 'intracluster_distance')[['id','web_name', 'intracluster_distance','now_cost','influence','average_fdr','minutes']]
    top_defenders = top_potential_players[top_potential_players['position'] == 'defender'].nlargest(15, 'intracluster_distance')[['id','web_name', 'intracluster_distance','now_cost','influence','average_fdr','minutes']]
    top_midfielders = top_potential_players[top_potential_players['position'] == 'midfielder'].nlargest(15, 'intracluster_distance')[['id','web_name', 'intracluster_distance','now_cost','influence','average_fdr','minutes']]
    top_forwards = top_potential_players[top_potential_players['position'] == 'forward'].nlargest(15, 'intracluster_distance')[['id','web_name', 'intracluster_distance','now_cost','influence','average_fdr','minutes']]

    return top_goalkeepers, top_defenders, top_midfielders, top_forwards, players_df

# Usage
top_goalkeepers, top_defenders, top_midfielders, top_forwards, all_players = get_top_players_by_position()

# Display results
print("Top Goalkeepers:")
print(top_goalkeepers)

print("\nTop Defenders:")
print(top_defenders)

print("\nTop Midfielders:")
print(top_midfielders)

print("\nTop Forwards:")
print(top_forwards)

# Fetch intracluster distance for a specific player, for example, "Salah"
# intracluster_distance = get_intracluster_distance('M.Salah')
# print (intracluster_distance)