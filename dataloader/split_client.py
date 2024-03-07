import numpy as np
import pdb

def split_clients(num_clients, num_users_per_client, train_data, val_data, test_data, user_count):
    dict_clients = {}

    # distribute user data to different clients
    counter = 0
    all_user_ids = [i for i in range(1, user_count)] # this is all indices
    all_user_ids = set(all_user_ids)
    for i in range(num_clients):
        # set a random seed for each user to determine the distribution for reproducibiliity
        np.random.seed(i)
        selected_users = np.random.choice(list(all_user_ids), num_users_per_client, replace=False)
        all_user_ids = set(all_user_ids) - set(selected_users)
        all_user_ids = sorted(list(all_user_ids))
        
        train_data_per_client = {}
        val_data_per_client = {}
        for current_user in selected_users:
            train_data_per_client[current_user] = train_data[current_user] + val_data[current_user]
            val_data_per_client[current_user] = test_data[current_user]

        dict_clients[i] = [train_data_per_client, val_data_per_client]

    test_client = {}
    for current_user in all_user_ids:
        test_client[current_user] = [train_data[current_user] + val_data[current_user], test_data[current_user]]

    return dict_clients, test_client