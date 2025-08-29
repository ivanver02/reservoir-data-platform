import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics.pairwise import haversine_distances

# Setting up the path to include the parent directory
sys.path.append(str(Path.cwd().parent.parent))
from backend.config.settings import PATHS, TRANSFER_OPTIMIZATION_THRESHOLDS
from backend.transferring.predicting import predict_one_year, predict_one_year_sarima
from backend.data.extract import extract_water_definitive, extract_reservoirs_merged_definitive

critical_threshold = TRANSFER_OPTIMIZATION_THRESHOLDS['critical']
worrying_threshold = TRANSFER_OPTIMIZATION_THRESHOLDS['worrying']
could_give_if_critical = TRANSFER_OPTIMIZATION_THRESHOLDS['could_give_if_critical']
could_give_if_worrying = TRANSFER_OPTIMIZATION_THRESHOLDS['could_give_if_worrying']
able_to_donate = TRANSFER_OPTIMIZATION_THRESHOLDS['able_to_donate']
cost_threshold = TRANSFER_OPTIMIZATION_THRESHOLDS['cost_threshold']
epsilon = TRANSFER_OPTIMIZATION_THRESHOLDS['epsilon']

water_pd = extract_water_definitive()
reservoirs_pd = extract_reservoirs_merged_definitive()

# Auxiliary function for initial_state
def create_water_with_predictions(reservoirs_list, series_index, water_restricted, plotting=False):
    water_definitive = pd.DataFrame()
    for reservoir in reservoirs_list:
        water_one_reservoir = pd.DataFrame()
        water_one_reservoir.loc[:, 'date'] = series_index
        water_one_reservoir.loc[water_one_reservoir['date'] < water_one_reservoir['date'][52], 'storage'] = water_restricted[water_restricted['id'] == reservoir].sort_values(by='date')['storage'][-52:].values
        try:
            water_one_reservoir.loc[water_one_reservoir['date'] >= water_one_reservoir['date'][52], 'storage'] = predict_one_year(reservoir, 4, plotting=plotting).values
        except Exception as e:
            water_one_reservoir.loc[water_one_reservoir['date'] >= water_one_reservoir['date'][52], 'storage'] = predict_one_year_sarima(reservoir).values
        water_one_reservoir.loc[:, 'id'] = reservoir
        water_definitive = pd.concat([water_definitive, water_one_reservoir], axis=0, ignore_index=True)

    water_definitive.sort_values(by=['id', 'date'], inplace=True)
    return water_definitive

# Auxiliary function for initial_state
def feature_engineering_reservoirs(reservoirs_restricted, reservoirs_list, water_definitive):
    reservoirs_definitive = reservoirs_restricted[['id', 'name', 'capacity', 'latitude', 'longitude', 'crest_elevation']].copy()
    reservoirs_definitive.set_index('id', inplace=True)
    for reservoir in reservoirs_list:
        reservoirs_definitive.loc[reservoir, 'last_known_value'] = water_definitive[water_definitive['id'] == reservoir].sort_values(by='date')['storage'].values[-53]
        reservoirs_definitive.loc[reservoir, 'low_forecasted_value'] = water_definitive[water_definitive['id'] == reservoir][-52:].sort_values(by='storage')['storage'].values[5]
        reservoirs_definitive.loc[reservoir, 'high_forecasted_value'] = water_definitive[water_definitive['id'] == reservoir][-52:].sort_values(by='storage')['storage'].values[-5]
        reservoirs_definitive.loc[reservoir, 'median_forecasted_value'] = water_definitive[water_definitive['id'] == reservoir][-52:].sort_values(by='storage')['storage'].values[26]

    reservoirs_definitive.loc[:, 'urgent_need_to_receive_rate'] = reservoirs_definitive['low_forecasted_value'] / reservoirs_definitive['capacity']
    reservoirs_definitive.loc[:, 'ability_to_transfer_rate'] = reservoirs_definitive['median_forecasted_value']  / reservoirs_definitive['capacity']
    reservoirs_definitive.loc[:, 'critical'] = (reservoirs_definitive['urgent_need_to_receive_rate'] < critical_threshold).astype(int)
    reservoirs_definitive.loc[:, 'worrying'] = (reservoirs_definitive['urgent_need_to_receive_rate'] < worrying_threshold).astype(int)
    reservoirs_definitive.loc[:, 'could_give_if_critical'] = (reservoirs_definitive['ability_to_transfer_rate'] > could_give_if_critical).astype(int)
    reservoirs_definitive.loc[:, 'could_give_if_worrying'] = (reservoirs_definitive['ability_to_transfer_rate'] > could_give_if_worrying).astype(int)
    reservoirs_definitive.loc[:, 'able_to_donate'] = (reservoirs_definitive['ability_to_transfer_rate'] > able_to_donate).astype(int)
    return reservoirs_definitive

# Auxiliary function for initial_state
def calculate_cost_matrix(reservoirs_definitive):
    reservoirs = sorted(reservoirs_definitive.index)
    coords_rad = np.radians(reservoirs_definitive[['latitude','longitude']].to_numpy())
    distance_matrix = haversine_distances(coords_rad, coords_rad) * 6371.0  # shape (N,N) in km
    # cost_matrix[i,j] means the cost of reservoir i to transfer water to reservoir j
    value_matrix = distance_matrix.copy()
    value_matrix = value_matrix / value_matrix.max()  # Normalize distances by the maximum distance in each row
    value_matrix = 4*value_matrix
    value_matrix = 1 / (value_matrix + epsilon)

    median_capacity = np.median(reservoirs_definitive['capacity'].values)

    for i in range(len(reservoirs)):
        # The next operation only depends on the donor reservoir's characteristics, not the receiver
        value_matrix[i, :] = value_matrix[i, :] * (reservoirs_definitive.loc[reservoirs[i], 'capacity'] / median_capacity) * (reservoirs_definitive.loc[reservoirs[i], 'ability_to_transfer_rate'] + epsilon * 1e4)
        # The next operation only depends on the receiver reservoir's characteristics, not the donor
        value_matrix[:, i] = value_matrix[:, i] / (reservoirs_definitive.loc[reservoirs[i], 'capacity'] / median_capacity * (reservoirs_definitive.loc[reservoirs[i], 'urgent_need_to_receive_rate'] + epsilon * 1e4))

    cost_matrix = 1 / value_matrix

    for i  in range(len(reservoirs)):
        cost_matrix[i,i] = np.inf
        if not (reservoirs_definitive.loc[reservoirs[i], 'could_give_if_critical']):
            cost_matrix[i,:] = np.full(len(reservoirs), np.inf)

    return cost_matrix, distance_matrix

# First important function
def initial_state(reservoirs_list, plotting=False, critical_threshold=critical_threshold, worrying_threshold=worrying_threshold, could_give_if_critical=could_give_if_critical, could_give_if_worrying=could_give_if_worrying, able_to_donate=able_to_donate):
    water_restricted = water_pd[water_pd['id'].isin(reservoirs_list)]
    reservoirs_restricted = reservoirs_pd[reservoirs_pd['id'].isin(reservoirs_list)]

    first_reservoir_data = water_restricted[water_restricted['id']==reservoirs_list[0]].sort_values(by='date')
    first_week_considered = first_reservoir_data.iloc[-52]['date']
    series_index = pd.date_range(start=first_week_considered, periods=52*2, freq='7D')

    water_definitive = create_water_with_predictions(reservoirs_list, series_index, water_restricted, plotting=plotting)
    reservoirs_definitive = feature_engineering_reservoirs(reservoirs_restricted, reservoirs_list, water_definitive)
    cost_matrix, distance_matrix = calculate_cost_matrix(reservoirs_definitive)

    return reservoirs_definitive, cost_matrix, distance_matrix


# Auxiliary function for update_reservoirs_equal_percentage
def calculate_difference(reservoirs, donor, receiver):
    capacity_donor = reservoirs.loc[donor, 'capacity']
    capacity_receiver = reservoirs.loc[receiver, 'capacity']
    current_storage_donor = reservoirs.loc[donor, 'last_known_value']
    current_storage_receiver = reservoirs.loc[receiver, 'last_known_value']
    ponderation_donor = capacity_donor / (capacity_donor + capacity_receiver)
    difference = current_storage_donor - ponderation_donor * (current_storage_donor + current_storage_receiver)
    return difference

def update_reservoirs_equal_percentage(reservoirs_definitive, donor, receiver):
    '''
    Updates the reservoirs by transferring water from the donor to the receiver
    in a way that maintains the same percentage of water in both reservoirs.

    Doesn't change the given reservoir.
    '''
    reservoirs = reservoirs_definitive.copy()
    difference = calculate_difference(reservoirs, donor, receiver)

    def update_one_reservoir(reservoir, difference):
        reservoirs.loc[reservoir, ['last_known_value', 'low_forecasted_value', 'high_forecasted_value', 'median_forecasted_value']] += difference
        reservoirs.loc[reservoir, 'urgent_need_to_receive_rate'] = reservoirs.loc[reservoir, 'low_forecasted_value'] / reservoirs.loc[reservoir, 'capacity']
        reservoirs.loc[reservoir, 'ability_to_transfer_rate'] = reservoirs.loc[reservoir, 'median_forecasted_value'] / reservoirs.loc[reservoir, 'capacity']
        reservoirs.loc[reservoir, 'critical'] = (reservoirs.loc[reservoir, 'urgent_need_to_receive_rate'] < critical_threshold).astype(int)
        reservoirs.loc[reservoir, 'worrying'] = (reservoirs.loc[reservoir, 'urgent_need_to_receive_rate'] < worrying_threshold).astype(int)
        reservoirs.loc[reservoir, 'could_give_if_critical'] = (reservoirs.loc[reservoir, 'ability_to_transfer_rate'] > could_give_if_critical).astype(int)
        reservoirs.loc[reservoir, 'could_give_if_worrying'] = (reservoirs.loc[reservoir, 'ability_to_transfer_rate'] > could_give_if_worrying).astype(int)
        reservoirs.loc[reservoir, 'able_to_donate'] = (reservoirs.loc[reservoir, 'ability_to_transfer_rate'] > able_to_donate).astype(int)

    update_one_reservoir(donor, -difference)
    update_one_reservoir(receiver, difference)

    return reservoirs, difference


# Important function to simulate transfers
def perform_best_transfer(reservoirs_definitive, cost_matrix, cost_threshold, accept_worrying=True):
    '''
    Performs the best transfer that fixes a critical, and if not possible, a worrying reservoir, following the costs.

    Aspects
    --------
    - It gives priority to critical receivers over worrying ones.
    - By default, a critical transfer whose involved reservoirs may end up as worrying (but neither of them critical), is accepted. If this is not desired, set accept_worrying to False.

    Returns
    -------
    - A DataFrame with the updated reservoirs if a valid transfer is found, otherwise the original DataFrame.
    - A boolean that is True if the given reservoir has been modified.
    '''
    def perform_best_transfer_for_given_columns(columns, state_receiver):
        i = 0
        index_to_id = {i: id for i, id in enumerate(reservoirs_definitive.index)}
        sorted_costs = np.sort(cost_matrix[:, columns], axis=None)
        while i < len(sorted_costs) and sorted_costs[i] < cost_threshold:
            # Get the indices of the reservoirs involved in this transfer
            reservoir_pair = np.where(cost_matrix[:, columns] == sorted_costs[i])
            donor = reservoir_pair[0][0]
            receiver = reservoir_pair[1][0]
            donor = index_to_id[donor]
            receiver = index_to_id[columns[receiver]]
            if reservoirs_definitive.loc[donor, f'could_give_if_{state_receiver}']:

                temptative_reservoirs_definitive, difference = update_reservoirs_equal_percentage(reservoirs_definitive, donor, receiver)
                transfer = {'donor': donor, 'receiver': receiver, 'difference': difference, 'cost': sorted_costs[i],
                                           'critical': temptative_reservoirs_definitive.loc[receiver, 'critical'],
                                           'worrying': temptative_reservoirs_definitive.loc[receiver, 'worrying'],
                                           'could_give_if_critical': temptative_reservoirs_definitive.loc[donor, 'could_give_if_critical'],
                                           'could_give_if_worrying': temptative_reservoirs_definitive.loc[donor, 'could_give_if_worrying'],
                                           'able_to_donate': temptative_reservoirs_definitive.loc[donor, 'able_to_donate']}

                if state_receiver == 'critical':
                    if (not temptative_reservoirs_definitive.loc[receiver, 'worrying']) and (not temptative_reservoirs_definitive.loc[donor, 'worrying']):
                        return temptative_reservoirs_definitive, True, transfer
                    elif accept_worrying and (not temptative_reservoirs_definitive.loc[receiver, 'critical']) and (not temptative_reservoirs_definitive.loc[donor, 'critical']):
                        return temptative_reservoirs_definitive, True, transfer
                    else:
                        i += 1
                        continue
                
                if state_receiver == 'worrying':
                    if (not temptative_reservoirs_definitive.loc[receiver, 'worrying']) and (not temptative_reservoirs_definitive.loc[donor, 'worrying']):
                        return temptative_reservoirs_definitive, True, transfer
                    else:
                        i += 1

            else:
                i += 1
        return pd.DataFrame(), False, None

    id_to_index = {id: i for i, id in enumerate(reservoirs_definitive.index)}
    critical_indices = np.sort(reservoirs_definitive[reservoirs_definitive['critical']==1].index.map(id_to_index).values)
    temptative_reservoirs_definitive, bool, transfer = perform_best_transfer_for_given_columns(critical_indices, 'critical')
    if bool:
        return temptative_reservoirs_definitive, True, transfer
    else:
        worrying_indices = np.sort(reservoirs_definitive[reservoirs_definitive['worrying']==1].index.map(id_to_index).values)
        worrying_indices = [i for i in worrying_indices if i not in critical_indices]
        temptative_reservoirs_definitive, bool, transfer = perform_best_transfer_for_given_columns(worrying_indices, 'worrying')
        if bool:
            return temptative_reservoirs_definitive, True, transfer
        else:
            return reservoirs_definitive, False, transfer


# Important function if only information is needed
def show_best_transfers(reservoirs, cost_matrix):
    '''
    Shows the best possible transfers but segmented into three groups:
        - Critical receiver
        - Worrying receiver
        - Rest of the receivers
    Each of them ordered by cost
    '''
    def show_best_transfers_for_given_columns(columns):
        i = 0
        index_to_id = {i: id for i, id in enumerate(reservoirs.index)}
        sorted_costs = np.sort(cost_matrix[:, columns], axis=None)
        transfers = pd.DataFrame(columns=['donor', 'receiver', 'difference', 'cost', 'critical', 'worrying', 'could_give_if_critical', 'could_give_if_worrying', 'able_to_donate'])
        while i < len(sorted_costs) and sorted_costs[i] < cost_threshold:
            # Get the indices of the reservoirs involved in this transfer
            reservoir_pair = np.where(cost_matrix[:, columns] == sorted_costs[i])
            donor = reservoir_pair[0][0]
            receiver = reservoir_pair[1][0]
            donor = index_to_id[donor]
            receiver = index_to_id[columns[receiver]]

            difference = calculate_difference(reservoirs, donor, receiver)

            transfers.loc[len(transfers)] = {'donor': donor, 'receiver': receiver, 'difference': difference, 'cost': sorted_costs[i],
                                           'critical': reservoirs.loc[receiver, 'critical'],
                                           'worrying': reservoirs.loc[receiver, 'worrying'],
                                           'could_give_if_critical': reservoirs.loc[donor, 'could_give_if_critical'],
                                           'could_give_if_worrying': reservoirs.loc[donor, 'could_give_if_worrying'],
                                           'able_to_donate': reservoirs.loc[donor, 'able_to_donate']}
            i += 1

        return transfers
    

    id_to_index = {id: i for i, id in enumerate(reservoirs.index)}
    critical_indices = reservoirs[reservoirs['critical']==1].index.map(id_to_index).values
    critical_transfers = show_best_transfers_for_given_columns(critical_indices)
    worrying_indices = reservoirs[reservoirs['worrying']==1].index.map(id_to_index).values
    worrying_indices = [i for i in worrying_indices if i not in critical_indices]
    worrying_transfers = show_best_transfers_for_given_columns(worrying_indices)
    rest_of_indices = [i for i in id_to_index.values() if i not in critical_indices and i not in worrying_indices]
    rest_of_transfers = show_best_transfers_for_given_columns(rest_of_indices)
    dfs_to_concat = [df for df in [critical_transfers, worrying_transfers, rest_of_transfers] if not df.empty]
    if dfs_to_concat:
        return pd.concat(dfs_to_concat, axis=0, ignore_index=True)
    else:
        return pd.DataFrame(columns=['donor', 'receiver', 'difference', 'cost', 'critical', 'worrying', 'could_give_if_critical', 'could_give_if_worrying', 'able_to_donate'])
    

def show_best_transfers_region(latitude, longitude, radius_km, cost_threshold=1.5, accept_worrying=True):
    '''
    Shows the best possible transfers between reservoirs located in a certain radius of action
    '''
    coord_rads = np.radians([latitude, longitude]).reshape(1, -1)
    reservoir_coords_rad = np.radians(reservoirs_pd[['latitude','longitude']].to_numpy())
    
    distances = haversine_distances(coord_rads, reservoir_coords_rad) * 6371.0  # shape (1,N) in km
    mask = distances[0] <= radius_km
    reservoirs = reservoirs_pd.loc[mask, 'id'].unique()
    reservoirs_pd.drop(columns='distance', inplace=True)

    reservoirs_definitive, cost_matrix, _ = initial_state(reservoirs)

    return show_best_transfers(reservoirs_definitive, cost_matrix, cost_threshold=cost_threshold, accept_worrying=accept_worrying)


def update_cost_matrix(donor, receiver, cost_matrix, distance_matrix, reservoirs_updated):

    reservoirs = sorted(reservoirs_updated.index)
    id_to_index = {id: i for i, id in enumerate(reservoirs)}
    Dmax = distance_matrix.max() if distance_matrix.size else 1.0
    median_capacity = np.median(reservoirs_updated['capacity'].values)

    # Pre-compute receiver factors & donor factors arrays for reuse when updating columns/rows
    def compute_receiver_factors():
        return np.array([
            (reservoirs_updated.loc[rid, 'capacity'] / median_capacity) *
            (reservoirs_updated.loc[rid, 'urgent_need_to_receive_rate'] + epsilon * 1e4)
            for rid in reservoirs
        ])

    def compute_donor_factors():
        return np.array([
            (reservoirs_updated.loc[rid, 'capacity'] / median_capacity) *
            (reservoirs_updated.loc[rid, 'ability_to_transfer_rate'] + epsilon * 1e4)
            for rid in reservoirs
        ])

    # These whole vectors are needed for each updated row/column (both donor & receiver updates)
    receiver_factors_all = compute_receiver_factors()
    donor_factors_all = compute_donor_factors()

    def recompute_row(i_idx, i_id):
        # donor i to every receiver j
        distances_row = distance_matrix[i_idx, :] / Dmax
        base_row = 1 / (4 * distances_row + epsilon)
        donor_factor_i = (reservoirs_updated.loc[i_id, 'capacity'] / median_capacity) * \
                         (reservoirs_updated.loc[i_id, 'ability_to_transfer_rate'] + epsilon * 1e4)
        value_row = base_row * donor_factor_i / receiver_factors_all
        cost_matrix[i_idx, :] = 1 / value_row

    def recompute_column(j_idx, j_id):
        # each donor i to receiver j
        distances_col = distance_matrix[:, j_idx] / Dmax
        base_col = 1 / (4 * distances_col + epsilon)
        receiver_factor_j = (reservoirs_updated.loc[j_id, 'capacity'] / median_capacity) * \
                             (reservoirs_updated.loc[j_id, 'urgent_need_to_receive_rate'] + epsilon * 1e4)
        value_col = base_col * donor_factors_all / receiver_factor_j
        cost_matrix[:, j_idx] = 1 / value_col

    # Update donor & receiver rows and columns
    donor_idx = id_to_index[donor]
    receiver_idx = id_to_index[receiver]
    recompute_row(donor_idx, donor)
    recompute_column(donor_idx, donor)
    recompute_row(receiver_idx, receiver)
    recompute_column(receiver_idx, receiver)

    # Apply constraints (self-transfer + giving eligibility)
    for rid, idx in id_to_index.items():
        cost_matrix[idx, idx] = np.inf
        if not reservoirs_updated.loc[rid, 'could_give_if_critical']:
            cost_matrix[idx, :] = np.inf

    return cost_matrix


# Main function
def make_optimal_transfers(reservoirs, cost_threshold=1.5, max_iterations=10, accept_worrying=True, plotting=False):
    '''
    Orchestrates the process of making optimal transfers between reservoirs.
    '''
    # Creates the initial state
    reservoirs_definitive, cost_matrix, distance_matrix = initial_state(reservoirs, plotting)

    iteration = 0
    transfer_done = True
    transfers_log = pd.DataFrame(columns=['donor', 'receiver', 'difference', 'cost', 'critical', 'worrying', 'could_give_if_critical', 'could_give_if_worrying', 'able_to_donate'])

    best_transfers_list = []

    while transfer_done and iteration < max_iterations:
        # Show the best transfers
        best_transfers = show_best_transfers(reservoirs_definitive, cost_matrix)

        if best_transfers.empty:
            break
        else:
            best_transfers_list.append(best_transfers)

        # Perform the best transfer
        reservoirs_definitive, transfer_done, transfer = perform_best_transfer(reservoirs_definitive, cost_matrix, cost_threshold, accept_worrying)
        if transfer_done:
            # Log the transfer
            transfers_log.loc[len(transfers_log)] = transfer
            # Update the cost matrix
            cost_matrix = update_cost_matrix(transfer['donor'], transfer['receiver'], cost_matrix, distance_matrix, reservoirs_definitive)

        iteration += 1

    return reservoirs_definitive, transfers_log


def make_optimal_transfers_region(latitude, longitude, radius_km, cost_threshold=1.5, max_iterations=10, accept_worrying=True):
    '''
    Orchestrates the process of making optimal transfers between reservoirs located in a certain radius of action
    '''
    coord_rads = np.radians([latitude, longitude]).reshape(1, -1)
    reservoir_coords_rad = np.radians(reservoirs_pd[['latitude','longitude']].to_numpy())
    
    reservoirs_pd['distance'] = (haversine_distances(coord_rads, reservoir_coords_rad) * 6371.0).reshape(-1)
    mask = reservoirs_pd['distance'] <= radius_km
    reservoirs = reservoirs_pd.loc[mask, 'id'].unique()
    reservoirs_pd.drop(columns='distance', inplace=True)

    return make_optimal_transfers(reservoirs, cost_threshold=cost_threshold, max_iterations=max_iterations, accept_worrying=accept_worrying)


# Plotting functions

def plot_all_reservoirs(reservoirs_list, with_transfers=False, legend=True):
    id_to_capacity = reservoirs_pd.set_index('id')['capacity'].to_dict()
    difference = {}
    transfers_log = pd.DataFrame()
    if with_transfers:
        
        _, transfers_log = make_optimal_transfers(reservoirs_list)

        for _, row in transfers_log.iterrows():
            if row['donor'] not in difference:
                difference[row['donor']] = 0
            difference[row['donor']] -= row['difference']
            if row['receiver'] not in difference:
                difference[row['receiver']] = 0
            difference[row['receiver']] += row['difference']

    next_year_predictions_path = PATHS['processed_data_notebooks'] / 'next_year_predictions.parquet'
    next_year_predictions = pd.read_parquet(next_year_predictions_path)

    water_pd['capacity'] = water_pd['id'].map(id_to_capacity)

    plt.figure(figsize=(12, 6))

    for reservoir in reservoirs_list:
        reservoir_pd = water_pd[water_pd['id'] == reservoir]
        reservoir_pd.set_index('date', inplace=True)
        df1 = reservoir_pd['storage'][-52*4:]/id_to_capacity[reservoir]
        df2 = (next_year_predictions[reservoir] + difference.get(reservoir, 0))/id_to_capacity[reservoir]
        df = pd.concat([df1, df2], axis=0)
        
        plt.plot(df.index,
                df,
                label=f'Reservoir {reservoir}')
        
    plt.axvline(x=df1.index[-1], color='k', linestyle='--', label='Next Year Prediction')
    plt.xlabel('Date')
    plt.ylabel('Storage (%)')
    plt.title('Evolution of Reservoir Storage')
    if legend:
        plt.legend()
    plt.show()
    return transfers_log

def plot_all_reservoirs_involved(reservoirs_list, with_transfers=False, legend=True):
    id_to_capacity = reservoirs_pd.set_index('id')['capacity'].to_dict()
    difference = {}
    _, transfers_log = make_optimal_transfers(reservoirs_list)

    for _, row in transfers_log.iterrows():
            if row['donor'] not in difference:
                difference[row['donor']] = 0
            difference[row['donor']] -= row['difference']
            if row['receiver'] not in difference:
                difference[row['receiver']] = 0
            difference[row['receiver']] += row['difference']
        
    reservoirs_involved = difference.keys()
    if not with_transfers:
         difference = {}

    next_year_predictions_path = PATHS['processed_data_notebooks'] / 'next_year_predictions.parquet'
    next_year_predictions = pd.read_parquet(next_year_predictions_path)

    water_pd['capacity'] = water_pd['id'].map(id_to_capacity)

    plt.figure(figsize=(12, 6))

    for reservoir in reservoirs_involved:
        reservoir_pd = water_pd[water_pd['id'] == reservoir]
        reservoir_pd.set_index('date', inplace=True)
        df1 = reservoir_pd['storage'][-52*4:]/id_to_capacity[reservoir]
        df2 = (next_year_predictions[reservoir] + difference.get(reservoir, 0))/id_to_capacity[reservoir]
        df = pd.concat([df1, df2], axis=0)
        
        plt.plot(df.index,
                df,
                label=f'Reservoir {int(reservoir)}')
        
    plt.axvline(x=df1.index[-1], color='k', linestyle='--', label='Next Year Prediction')
    plt.xlabel('Date')
    plt.ylabel('Storage (%)')
    plt.title('Evolution of Reservoir Storage')
    if legend:
        plt.legend()
    plt.show()
    return transfers_log

if __name__ == "__main__":
    
    reservoirs = sorted(reservoirs_pd[reservoirs_pd['province'] == 'cadiz']['id'].values)
    reservoirs_cadiz, cost_matrix_cadiz, distance_matrix_cadiz = initial_state(reservoirs, plotting=False)

    print(f"Reservoirs in Cadiz: {reservoirs_cadiz}")

    print(show_best_transfers(reservoirs_cadiz, cost_matrix_cadiz))

    final_reservoirs, transfers_log = make_optimal_transfers(reservoirs, cost_threshold=1.5, max_iterations=10, accept_worrying=True)

    print(transfers_log)
    '''
    reservoirs = sorted(reservoirs_pd[reservoirs_pd['province'] == 'cadiz']['id'].values)
    reservoirs_cadiz, cost_matrix_cadiz, distance_matrix_cadiz = initial_state(reservoirs)
    updated_reservoirs_cadiz, difference = update_reservoirs_equal_percentage(reservoirs_cadiz, 375, 212)
    calculated_cost_matrix, hola = calculate_cost_matrix(updated_reservoirs_cadiz)


    # Update cost matrix from the previous one
    coords_rad = np.radians(reservoirs_cadiz[['latitude','longitude']].to_numpy())
    updated_cost_matrix = update_cost_matrix(375, 212, cost_matrix_cadiz, distance_matrix_cadiz, updated_reservoirs_cadiz)


    print((calculated_cost_matrix - updated_cost_matrix))
    '''