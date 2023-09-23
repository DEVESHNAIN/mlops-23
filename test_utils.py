from utils import fetch_hyperparameter_combinations, split_train_dev_test,read_digits

def test_for_hparam_cominations_count():
    # a test case to check that all possible combinations of paramers are indeed generated
    gamma_values = [0.001, 0.01, 0.1, 1]
    C_values = [1, 10, 100, 1000]
    hyper_params={}
    hyper_params['gamma'] = gamma_values
    hyper_params['C'] = C_values
    hyper_params_comb = fetch_hyperparameter_combinations(hyper_params)
    
    assert len(hyper_params_comb) == len(gamma_values) * len(C_values)

def test_for_hparam_cominations_values():    
    gamma_values = [0.001, 0.01]
    C_values = [1]
    hyper_params={}
    hyper_params['gamma'] = gamma_values
    hyper_params['C'] = C_values
    hyper_params_comb = fetch_hyperparameter_combinations(hyper_params)

    expected_param_combo_1 = {'gamma': 0.001, 'C': 1}
    expected_param_combo_2 = {'gamma': 0.01, 'C': 1}

    assert (expected_param_combo_1 in hyper_params_comb) and (expected_param_combo_2 in hyper_params_comb)


def test_data_splitting():
    X, y = read_digits()
    
    X = X[:100,:,:]
    y = y[:100]
    
    test_size = .2
    dev_size = .2
    train_size = 1 - test_size - dev_size

    X_train, X_test, X_dev, y_train, y_test, y_dev = split_train_dev_test(X, y, test_size=test_size, dev_size=dev_size)

    assert (len(X_train) == 60) 
    assert (len(X_test) == 20)
    assert  ((len(X_dev) == 20))

