from flask import Flask, render_template, request, redirect, url_for
from functions import get_current_gameweek, get_input_squad, get_team_info, get_top_players_by_position, assign_bench_and_watch_avoid,suggest_multiple_transfers
from functions import select_starting_11_and_bench, get_player_data, get_free_transfers, suggest_transfer,sort_by_pca_scores,determine_captain_and_vice_captain
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def get_team_id():
    if request.method == 'POST':
        # Get team_id from the form submission
        team_id = request.form['team_id']
        return redirect(url_for('show_team_info', team_id=team_id))
    return render_template('get_team_id.html')

@app.route('/team/<team_id>', methods=['GET'])
def show_team_info(team_id):
    gw_number = get_current_gameweek()
    
    input_squad = get_input_squad(int(team_id), gw_number)
    team_info = get_team_info(int(team_id))

    players_df = get_player_data()

    top_goalkeepers, top_defenders, top_midfielders, top_forwards, all_players = get_top_players_by_position()

    if team_info is None:
        return "Error: Team information not found"

    # Extract required fields from team_info
    player_first_name = team_info.get('player_first_name', "")
    player_last_name = team_info.get('player_last_name', "")
    welcome_message = f"Welcome {player_first_name} {player_last_name}!"

    gameweek_message = f"For Gameweek {gw_number + 1}"

    # Call the function to get players to keep and watch/avoid
    keep, watch_avoid = assign_bench_and_watch_avoid(input_squad['id'])
    sorted_keep = keep.sort_values(by='pca_score', ascending=False)
      # Sort the 'watch_avoid' DataFrame by 'intracluster_distance' and display
    sorted_watch_avoid = sort_by_pca_scores(watch_avoid)

    # Merge sorted_watch_avoid and keep into input_squad_with_pca
    input_squad_with_pca = pd.concat([sorted_watch_avoid, keep], ignore_index=True)

    # Format the data to display
    # keep_players = sorted_keep[['id', 'web_name', 'pca_score']].rename(columns={'web_name':'Player Name', 'pca_score': 'score'}).to_dict(orient='records')

    # # Format Top Players
    # top_goalkeepers_list = top_goalkeepers[['web_name', 'pca_score', 'now_cost']].rename(columns={'web_name':'Player Name', 'pca_score': 'score', 'now_cost':'Cost'}).assign(Cost=lambda df: df['Cost'] / 10).to_dict(orient='records')
    # top_defenders_list = top_defenders[['web_name', 'pca_score', 'now_cost']].rename(columns={'web_name':'Player Name', 'pca_score': 'score', 'now_cost':'Cost'}).assign(Cost=lambda df: df['Cost'] / 10).to_dict(orient='records')
    # top_midfielders_list = top_midfielders[['web_name', 'pca_score', 'now_cost']].rename(columns={'web_name':'Player Name', 'pca_score': 'score', 'now_cost':'Cost'}).assign(Cost=lambda df: df['Cost'] / 10).to_dict(orient='records')
    # top_forwards_list = top_forwards[['web_name', 'pca_score', 'now_cost']].rename(columns={'web_name':'Player Name', 'pca_score': 'score', 'now_cost':'Cost'}).assign(Cost=lambda df: df['Cost'] / 10).to_dict(orient='records')

    # Convert DataFrames to lists of dictionaries
    top_goalkeepers_list = top_goalkeepers.to_dict(orient='records')
    top_defenders_list = top_defenders.to_dict(orient='records')
    top_midfielders_list = top_midfielders.to_dict(orient='records')
    top_forwards_list = top_forwards.to_dict(orient='records')


    starting_11, bench = select_starting_11_and_bench(input_squad, players_df)
    free_transfers_info = get_free_transfers(team_id)
    available_budget = free_transfers_info['last_deadline_bank'] / 10  # Convert to millions if needed
    free_transfers = free_transfers_info['available_transfers']

    suggested_watch_transfer = suggest_transfer(sorted_watch_avoid, top_goalkeepers,top_defenders,top_midfielders,top_forwards,available_budget,free_transfers,team_id)
    suggested_keep_transfer = suggest_transfer(keep, top_goalkeepers,top_defenders,top_midfielders,top_forwards,available_budget,5,team_id)
    multiple_transfer = suggest_multiple_transfers(input_squad_with_pca, top_goalkeepers,top_defenders,top_midfielders,top_forwards,available_budget,2,team_id)
    print(multiple_transfer)

    captain_and_vice = determine_captain_and_vice_captain(keep)    

    # Assuming you are using Flask or a similar framework
    return render_template('team_info.html',
                       welcome_message=f"Welcome {player_first_name}!",
                       gameweek_message=f"For Gameweek {gw_number + 1}",
                       starting_11=starting_11,
                       bench=bench,
                       available_budget=available_budget,
                       suggested_watch_transfer=suggested_watch_transfer,
                       suggested_keep_transfer = suggested_keep_transfer,
                       captain = captain_and_vice['captain'],
                       vice_captain = captain_and_vice['vice_captain'],
                       top_goalkeepers=top_goalkeepers_list,
                       top_defenders=top_defenders_list,
                       top_midfielders=top_midfielders_list,
                       top_forwards=top_forwards_list)  # Add suggested transfers data here


if __name__ == "__main__":
    app.run(debug=True)
