import pandas as pd
import requests
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors

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

    # Function to fetch players data
    def calculate_average_fdr(fixtures_df):
        # Filter for upcoming fixtures
        remaining_fixtures = fixtures_df[fixtures_df['finished'] == False]

        # Create a DataFrame to hold FDR values
        fdr_values = []
    
        for idx, fixture in remaining_fixtures.iterrows():
            team_h = fixture['team_h']
            team_a = fixture['team_a']
            fdr_h = fixture['team_h_difficulty']
            fdr_a = fixture['team_a_difficulty']
            
            # Append the home and away FDR to the list
            fdr_values.append({'team': team_h, 'fdr': fdr_h})
            fdr_values.append({'team': team_a, 'fdr': fdr_a})
    
        # Convert to DataFrame
        fdr_df = pd.DataFrame(fdr_values)
    
        # Calculate the average FDR for each team
        average_fdr = fdr_df.groupby('team')['fdr'].mean().reset_index()
        average_fdr.columns = ['team', 'average_fdr']
        
        return average_fdr

    # Fetch the data
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
        5: 'Burnley', 6: 'Chelsea', 7: 'Crystal Palace', 8: 'Everton',
        9: 'Fulham', 10: 'Liverpool', 11: 'Man City', 12: 'Man Utd',
        13: 'Newcastle', 14: 'Norwich', 15: 'Southampton', 16: 'Tottenham',
        17: 'Watford', 18: 'West Ham', 19: 'Wolves', 20: 'Leeds'
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
    X = players_df[feature_columns]
    X = X.apply(pd.to_numeric, errors='coerce').dropna()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply K-Means Clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    players_df['cluster'] = kmeans.fit_predict(X_scaled)

    # # Calculate intracluster distances
    # players_df['distance'] = np.linalg.norm(X_scaled - kmeans.cluster_centers_[players_df['cluster']], axis=1)

    #Calculate intracluster distances
    def calculate_intracluster_distance(cluster_data):
        """Calculate the average distance of each point to all other points in its cluster."""
        distances = np.linalg.norm(cluster_data[:, np.newaxis] - cluster_data, axis=2)  # Pairwise distances
        avg_distances = distances.mean(axis=1)  # Average distance to all other points
        return avg_distances
    
    # Step 9: Calculate intracluster distances for each cluster
    players_df['intracluster_distance'] = np.nan
    for cluster in players_df['cluster'].unique():
        cluster_data = X_scaled[players_df['cluster'] == cluster]
        intracluster_distances = calculate_intracluster_distance(cluster_data)
        players_df.loc[players_df['cluster'] == cluster, 'intracluster_distance'] = intracluster_distances

    # Step 10: Check the number of players in each cluster
    cluster_counts = players_df['cluster'].value_counts()
    
    # Get top 10 players from each cluster based on intracluster distance
    # top_potential_players = top_potential_players.nlargest(10, 'intracluster_distance')
    # top_not_potential_players = top_not_potential_players.nlargest(10, 'intracluster_distance')

    # Map positions
    position_mapping = {
        1: 'goalkeeper', 2: 'defender', 3: 'midfielder', 4: 'forward'
    }
    players_df['position'] = players_df['element_type'].map(position_mapping)

    # Step 11: Rank players based on their intracluster distance
    top_potential_players = players_df[players_df['cluster'] == 0]
    # top_not_potential_players = fpl_data[fpl_data['cluster'] == 1]

    # Prepare lists for top players by position based on intracluster distance
    top_goalkeepers = top_potential_players[top_potential_players['position'] == 'goalkeeper'].nlargest(15, 'intracluster_distance')[['id','web_name', 'intracluster_distance','now_cost']]
    top_defenders = top_potential_players[top_potential_players['position'] == 'defender'].nlargest(15, 'intracluster_distance')[['id','web_name', 'intracluster_distance','now_cost']]
    top_midfielders = top_potential_players[top_potential_players['position'] == 'midfielder'].nlargest(15, 'intracluster_distance')[['id','web_name', 'intracluster_distance','now_cost']]
    top_forwards = top_potential_players[top_potential_players['position'] == 'forward'].nlargest(15, 'intracluster_distance')[['id','web_name', 'intracluster_distance','now_cost']]

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

