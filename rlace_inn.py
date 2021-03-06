import numpy as np
import tqdm
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
import time
from torch.optim import SGD, Adam
import random
import sklearn
import FrEIA as Fr
import FrEIA.framework as Ff
import FrEIA.modules as Fm

EVAL_CLF_PARAMS = {"loss": "log", "tol": 1e-4, "iters_no_change": 15, "alpha": 1e-4, "max_iter": 25000}
EVAL_CLF_PARAMS_INN = {"hidden_layer_sizes": (100, ), "activation":"relu", "learning_rate_init": 0.0003, "alpha": 1e-4, "max_iter": 25000,"tol": 1e-4}
NUM_CLFS_IN_EVAL = 3 # change to 1 for large dataset / high dimensionality

def init_classifier():

    return SGDClassifier(loss=EVAL_CLF_PARAMS["loss"], fit_intercept=True, max_iter=EVAL_CLF_PARAMS["max_iter"], tol=EVAL_CLF_PARAMS["tol"], n_iter_no_change=EVAL_CLF_PARAMS["iters_no_change"],
                        n_jobs=32, alpha=EVAL_CLF_PARAMS["alpha"])

def init_classifier_inn():
    return MLPClassifier(hidden_layer_sizes=EVAL_CLF_PARAMS_INN["hidden_layer_sizes"], activation=EVAL_CLF_PARAMS_INN["activation"], learning_rate_init=EVAL_CLF_PARAMS_INN["learning_rate_init"], max_iter=EVAL_CLF_PARAMS_INN["max_iter"], alpha=EVAL_CLF_PARAMS_INN["alpha"], tol=EVAL_CLF_PARAMS_INN["tol"])

def symmetric(X):
    X.data = 0.5 * (X.data + X.data.T)
    return X

def apply_inn(inn, X, P, type="tensor"):
    if type != "tensor":
        X = torch.as_tensor(X, dtype=torch.float)
        P = torch.as_tensor(P, dtype=torch.float)
    output = inn.forward(X)[0]
    #print(type(output), type(P))
    output = output @ P
    #print(type(output))
    output = inn.forward(output, rev=True)[0] 
    #print(output.shape)
    #print("————————")
    return output if type == "tensor" else output.detach().numpy()

def get_score(X_train, y_train, X_dev, y_dev, P, rank):
    P_svd = get_projection(P, rank)
    
    loss_vals = []
    accs = []
    
    for i in range(NUM_CLFS_IN_EVAL):
        clf = init_classifier()
        clf.fit(X_train @ P_svd, y_train)
        y_pred = clf.predict_proba(X_dev@P_svd)
        loss = sklearn.metrics.log_loss(y_dev, y_pred)
        loss_vals.append(loss)
        accs.append(clf.score(X_dev@P_svd, y_dev))
        
    i = np.argmin(loss_vals)
    return loss_vals[i], accs[i]

def get_score_inn(X_train, y_train, X_dev, y_dev, P, rank, inn):
    P_svd = get_projection(P, rank)
    
    loss_vals = []
    accs = []
    
    for i in range(NUM_CLFS_IN_EVAL):
        clf = init_classifier()
        clf.fit(apply_inn(inn, X_train, P_svd, type="numpy"), y_train)
        y_pred = clf.predict_proba(apply_inn(inn, X_dev,P_svd, type="numpy"))
        loss = sklearn.metrics.log_loss(y_dev, y_pred)
        loss_vals.append(loss)
        accs.append(clf.score(apply_inn(inn, X_dev,P_svd, type="numpy"), y_dev))
        
    i = np.argmin(loss_vals)
    return loss_vals[i], accs[i]


def solve_constraint(lambdas, d=1):
    def f(theta):
        return_val = np.sum(np.minimum(np.maximum(lambdas - theta, 0), 1)) - d
        return return_val

    theta_min, theta_max = max(lambdas), min(lambdas) - 1
    assert f(theta_min) * f(theta_max) < 0

    mid = (theta_min + theta_max) / 2
    tol = 1e-4
    iters = 0

    while iters < 25:

        mid = (theta_min + theta_max) / 2

        if f(mid) * f(theta_min) > 0:

            theta_min = mid
        else:
            theta_max = mid
        iters += 1

    lambdas_plus = np.minimum(np.maximum(lambdas - mid, 0), 1)
    # if (theta_min-theta_max)**2 > tol:
    #    print("didn't converge", (theta_min-theta_max)**2)
    return lambdas_plus

def get_majority_acc(y):

    from collections import Counter
    c = Counter(y)
    fracts = [v / sum(c.values()) for v in c.values()]
    maj = max(fracts)
    return maj

def get_entropy(y):

    from collections import Counter
    import scipy
    
    c = Counter(y)
    fracts = [v / sum(c.values()) for v in c.values()]
    return scipy.stats.entropy(fracts)
    

def get_projection(P, rank):
    #Singular value decomposition for dimensionality reduction
    D,U = np.linalg.eigh(P)
    U = U.T
    W = U[-rank:]
    P_final = np.eye(P.shape[0]) - W.T @ W
    return P_final

def prepare_output(P,rank,score):
    P_final = get_projection(P,rank)
    return {"score": score, "P_before_svd": np.eye(P.shape[0]) - P, "P": P_final}

def prepare_output_inn(P,rank,inn,score):
    P_final = get_projection(P,rank)
    return {"score": score, "P_before_svd": np.eye(P.shape[0]) - P, "P": P_final, "INN":inn}

def solve_adv_game(X_train, y_train, X_dev, y_dev, rank=1, device="cpu", out_iters=75000, in_iters_adv=1, in_iters_clf=1, epsilon=0.0015, batch_size=128, evalaute_every=1000, optimizer_class=SGD, 
optimizer_params_P={"lr": 0.005, "weight_decay": 1e-4}, optimizer_params_predictor={"lr": 0.005, "weight_decay": 1e-4}, optimizer_params_INN = {"lr": 0.005, "weight_decay": 1e-4}):
    """

    :param X: The input (np array)
    :param Y: the lables (np array)
    :param X_dev: Dev set (np array)
    :param Y_dev: Dev labels (np array)
    :param rank: Number of dimensions to neutralize from the input.
    :param device:
    :param out_iters: Number of batches to run
    :param in_iters_adv: number of iterations for adversary's optimization
    :param in_iters_clf: number of iterations from the predictor's optimization
    :param epsilon: stopping criterion .Stops if abs(acc - majority) < epsilon.
    :param batch_size:
    :param evalaute_every: After how many batches to evaluate the current adversary.
    :param optimizer_class: SGD/Adam etc.
    :param optimizer_params: the optimizer's params (as a dict)
    :return:
    """

    #subject to modification
    def get_loss_fn(X, y, predictor, P, bce_loss_fn, optimize_P=False):
        I = torch.eye(X_train.shape[1]).to(device)
        bce = bce_loss_fn(predictor(X @ (I - P)).squeeze(), y)
        if optimize_P:
            bce = -bce
        return bce

    def get_loss_fn_inn(X, y, predictor, P, inn, bce_loss_fn, optimize_P=False):
        I = torch.eye(X_train.shape[1]).to(device)
        bce = bce_loss_fn(predictor(apply_inn(inn, X, I-P)).squeeze(), y)
        if optimize_P:
            bce = -bce
        return bce

    def subnet_fc(c_in, c_out):
        return torch.nn.Sequential(torch.nn.Linear(c_in, 512), torch.nn.ReLU(),
                        torch.nn.Linear(512,  c_out))

    inn = Ff.SequenceINN(dim)
    for k in range(8):
        inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
    X_torch = torch.tensor(X_train).float().to(device)
    y_torch = torch.tensor(y_train).float().to(device)


    num_labels = len(set(y_train.tolist()))
    
    if num_labels == 2:
        predictor = torch.nn.Linear(X_train.shape[1], 1).to(device)
#
        bce_loss_fn = torch.nn.BCEWithLogitsLoss()
        y_torch = y_torch.float()
    else:
        predictor = torch.nn.Linear(X_train.shape[1], num_labels).to(device)
        #predictor = inn.append(predictor)
        bce_loss_fn = torch.nn.CrossEntropyLoss()
        y_torch = y_torch.long()

    #reconstruction loss
    #recons_loss = torch.nn.MSELoss()



    P = 1e-1*torch.randn(X_train.shape[1], X_train.shape[1]).to(device)
    P.requires_grad = True

    optimizer_predictor = optimizer_class(predictor.parameters(), **optimizer_params_predictor)
    optimizer_P = optimizer_class([P],**optimizer_params_P)
    optimizer_INN = optimizer_class(inn.parameters(),**optimizer_params_INN)

    maj = get_majority_acc(y_train)
    label_entropy = get_entropy(y_train)
    pbar = tqdm.tqdm(range(out_iters), total = out_iters, ascii=True)
    count_examples = 0
    best_P, best_score, best_loss = None, 1, -1

    for i in pbar:

        for j in range(in_iters_adv):

            idx = np.arange(0, X_torch.shape[0])
            np.random.shuffle(idx)
            X_batch, y_batch = X_torch[idx[:batch_size]], y_torch[idx[:batch_size]]
            P = symmetric(P)

            #train the INN using reconstruction loss
            optimizer_INN.zero_grad()
            recons_loss = torch.nn.MSELoss()
            loss_INN = recons_loss(X_batch, apply_inn(inn, X_batch, P))
            loss_INN.backward()
            optimizer_INN.step()

            
            optimizer_P.zero_grad()


            loss_P = get_loss_fn_inn(X_batch, y_batch, predictor, symmetric(P), inn, bce_loss_fn, optimize_P=True)
            loss_P.backward()
            optimizer_P.step()

            # project

            with torch.no_grad():
                D, U = torch.linalg.eigh(symmetric(P).detach().cpu())
                D = D.detach().cpu().numpy()
                D_plus_diag = solve_constraint(D, d=rank)
                D = torch.tensor(np.diag(D_plus_diag).real).float().to(device)
                U = U.to(device)
                P.data = U @ D @ U.T

        for j in range(in_iters_clf):
            optimizer_predictor.zero_grad()
            idx = np.arange(0, X_torch.shape[0])
            np.random.shuffle(idx)
            X_batch, y_batch = X_torch[idx[:batch_size]], y_torch[idx[:batch_size]]

            loss_predictor = get_loss_fn_inn(X_batch, y_batch, predictor, symmetric(P), inn, bce_loss_fn, optimize_P=False)
            loss_predictor.backward()
            optimizer_predictor.step()
            count_examples += batch_size

        if i % evalaute_every == 0:
            #pbar.set_description("Evaluating current adversary...")
            loss_val, score = get_score_inn(X_train, y_train, X_train, y_train, P.detach().cpu().numpy(), rank, inn)
            if loss_val > best_loss:#if np.abs(score - maj) < np.abs(best_score - maj):
                best_P, best_loss, best_inn = symmetric(P).detach().cpu().numpy().copy(), loss_val, inn
            if np.abs(score - maj) < np.abs(best_score - maj):
                best_score = score
                
            # update progress bar
            
            best_so_far = best_score if np.abs(best_score-maj) < np.abs(score-maj) else score
            
            pbar.set_description("{:.0f}/{:.0f}. Acc post-projection: {:.3f}%; best so-far: {:.3f}%; Maj: {:.3f}%; Gap: {:.3f}%; best loss: {:.4f}; current loss: {:.4f}".format(i, out_iters, score * 100, best_so_far * 100, maj * 100, np.abs(best_so_far - maj) * 100, best_loss, loss_val))
            pbar.refresh()  # to show immediately the update
            time.sleep(0.01)

        if i > 1 and np.abs(best_score - maj) < epsilon:
        #if i > 1 and np.abs(best_loss - label_entropy) < epsilon:
                    break
    output = prepare_output_inn(best_P,rank,best_inn,best_score)
    return output

if __name__ == "__main__":
    
    #random.seed(0)
    #np.random.seed(0)

    # create a synthetic dataset
    n, dim = 15000, 200
    num_classes = 2
    
    X = np.random.randn(n, dim)
    y = np.random.randint(low = 0, high = num_classes, size = n) #(np.random.rand(n) > 0.5).astype(int)

    X[:, 0] = (y + np.random.randn(*y.shape) * 0.3) ** 2 + 0.3 * y
    X[:, 1] = (y + np.random.randn(*y.shape) * 0.1) ** 2 - 0.7 * y
    X[:, 2] = (y + np.random.randn(*y.shape) * 0.3) ** 2 + 0.5 * y + np.random.randn(*y.shape) * 0.2
    X[:, 3] = (y + np.random.randn(*y.shape) * 0.5) ** 2 - 0.7 * y + np.random.randn(*y.shape) * 0.1
    X[:, 4] = (y + np.random.randn(*y.shape) * 0.5) ** 2 - 0.8 * y + np.random.randn(*y.shape) * 0.1
    X[:, 5] = (y + np.random.randn(*y.shape) * 0.25) ** 2 - 0.2 * y + np.random.randn(*y.shape) * 0.1
    mixing_matrix = 1e-2*np.random.randn(dim, dim)
    X = X @ mixing_matrix
    
    l_train = int(0.6*n)
    X_train, y_train = X[:l_train], y[:l_train]
    X_dev, y_dev = X[l_train:], y[l_train:]

    # arguments
    num_iters = 50000
    rank=1
    optimizer_class = torch.optim.SGD
    optimizer_params_INN = {"lr": 0.003, "weight_decay": 1e-4}
    optimizer_params_P = {"lr": 0.003, "weight_decay": 1e-4}
    optimizer_params_predictor = {"lr": 0.003,"weight_decay": 1e-4}
    epsilon = 0.001 # stop 0.1% from majority acc
    batch_size = 256


    output = solve_adv_game(X_train, y_train, X_dev, y_dev, rank=rank, device="cpu", out_iters=num_iters, optimizer_class=optimizer_class, optimizer_params_P =optimizer_params_P, optimizer_params_predictor=optimizer_params_predictor,optimizer_params_INN =optimizer_params_INN, epsilon=epsilon,batch_size=batch_size)
    
    # train a classifier
    
    P_svd = output["P"]
    inn = output["INN"]
    P_before_svd = output["P_before_svd"]
    svm = init_classifier_inn()
                        
    svm.fit(X_train[:] , y_train[:])
    score_original = svm.score(X_dev, y_dev)
    
    svm = init_classifier_inn()
    svm.fit(apply_inn(inn, X_train[:], P_before_svd, type="numpy") , y_train[:])
    score_projected_no_svd = svm.score(apply_inn(inn, X_dev, P_before_svd, type="numpy"), y_dev)
    
    svm = init_classifier_inn()
    svm.fit(apply_inn(inn, X_train[:], P_svd, type="numpy") , y_train[:])
    score_projected_svd_dev = svm.score(apply_inn(inn, X_dev, P_svd, type="numpy"), y_dev)
    score_projected_svd_train = svm.score(apply_inn(inn, X_train, P_svd, type="numpy"), y_train)
    maj_acc_dev = get_majority_acc(y_dev)
    maj_acc_train = get_majority_acc(y_train)
    
    print("===================================================")
    print("Original Acc, dev: {:.3f}%; Acc, projected, no svd, dev: {:.3f}%; Acc, projected+SVD, train: {:.3f}%; Acc, projected+SVD, dev: {:.3f}%".format(
        score_original*100, score_projected_no_svd*100, score_projected_svd_train*100, score_projected_svd_dev*100))    
    print("Majority Acc, dev: {:.3f} %".format(maj_acc_dev*100))
    print("Majority Acc, train: {:.3f} %".format(maj_acc_train*100))
    print("Gap, dev: {:.3f} %".format(np.abs(maj_acc_dev - score_projected_svd_dev)*100))
    print("Gap, train: {:.3f} %".format(np.abs(maj_acc_train - score_projected_svd_train)*100))
    print("===================================================")
    eigs_before_svd, _ = np.linalg.eigh(P_before_svd)
    print("Eigenvalues, before SVD: {}".format(eigs_before_svd[:]))
    
    eigs_after_svd, _ = np.linalg.eigh(P_svd)
    print("Eigenvalues, after SVD: {}".format(eigs_after_svd[:]))
    
    eps = 1e-6
    assert np.abs( (eigs_after_svd > eps).sum() -  (dim - rank) ) < eps
    
