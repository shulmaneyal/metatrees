import torch
from torch import nn

from batch.utils import weighted_mean


class Leaf(nn.Module):
    def __init__(self, preds):
        super(Leaf, self).__init__()
        self.register_buffer('preds', preds)

    @staticmethod
    def is_leaf():
        return True

    def predict(self, x):
        n_samples = x.shape[1]
        return self.preds.unsqueeze(1).repeat(1, n_samples, 1)

    def predict_hard(self, x):
        return self.predict(x)


class Tree(nn.Module):
    def __init__(self, weights, biases, beta, right, left, dist_func, temperature, stop=None,
                 leaf=None):
        super(Tree, self).__init__()

        self.register_buffer('rweights', weights)
        self.register_buffer('biases', biases)
        self.register_buffer('beta', beta)
        dist_weights = dist_func(weights * temperature)
        self.register_buffer('weights', dist_weights)
        self.right = right
        self.left = left
        if stop is None:
            stop = torch.zeros_like(biases)
        self.stop = stop
        if leaf is None:
            leaf = 0
        self.leaf = leaf

    @staticmethod
    def is_leaf():
        return False

    def get_right_probs(self, x):
        outs = torch.einsum("abc,ac->ab", (x, self.weights)) + self.biases
        right_probs = torch.sigmoid(outs * self.beta).unsqueeze(-1)
        return right_probs

    def predict(self, x):
        right_child_probs = self.right.predict(x)
        left_child_probs = self.left.predict(x)

        right_probs = self.get_right_probs(x)
        left_probs = 1 - right_probs

        my_probs = left_probs * left_child_probs + right_probs * right_child_probs
        return my_probs

    def predict_hard(self, x):
        right_child_probs = self.right.predict_hard(x)
        left_child_probs = self.left.predict_hard(x)

        right_probs = self.get_right_probs(x).round()
        left_probs = 1 - right_probs

        my_probs = left_probs * left_child_probs + right_probs * right_child_probs
        return my_probs


class Encoder(nn.Module):
    def __init__(self, x_dim, r_dim, y_dim):
        super(Encoder, self).__init__()
        self.l1 = nn.Linear(x_dim + y_dim, r_dim)
        self.l2 = nn.Linear(r_dim, r_dim)
        self.l3 = nn.Linear(r_dim, r_dim)
        self.l4 = nn.Linear(r_dim, r_dim)

    def forward(self, x, y):
        out = torch.cat((x, y), 2)
        out = self.l1(out)
        out = nn.functional.relu(out)
        out = self.l2(out)
        out = nn.functional.relu(out)
        out = self.l3(out)
        out = nn.functional.relu(out)
        out = self.l4(out)
        return out


class LeafBuilderClassification(nn.Module):
    def __init__(self, r_dim, y_dim):
        super(LeafBuilderClassification, self).__init__()
        self.l1 = nn.Linear(2 * r_dim, 20)
        self.l2 = nn.Linear(20, y_dim)

    def get_node(self, representation):
        leaf_preds = self.l1(representation)
        leaf_preds = nn.functional.relu(leaf_preds)
        leaf_preds = self.l2(leaf_preds)
        leaf_preds = nn.functional.softmax(leaf_preds, dim=1)
        return Leaf(leaf_preds)

    def forward(self, representations):
        pass


class LeafBuilderRegression(nn.Module):
    def __init__(self, r_dim, min_y, max_y):
        super(LeafBuilderRegression, self).__init__()
        self.min_y = min_y
        self.max_y = max_y
        self.l1 = nn.Linear(2 * r_dim, 20)
        self.l2 = nn.Linear(20, 1)

    def get_node(self, representation):
        leaf_preds = self.l1(representation)
        leaf_preds = nn.functional.relu(leaf_preds)
        leaf_preds = self.l2(leaf_preds)
        leaf_preds = self.min_y + (self.max_y - self.min_y) * torch.sigmoid(leaf_preds)
        return Leaf(leaf_preds)

    def forward(self, representations):
        pass


class TreeBuilder(nn.Module):
    def __init__(self, height, x_dim, r_dim, leaf_builder, dist_func, temperature):
        super(TreeBuilder, self).__init__()
        self.l1 = nn.Linear(2 * r_dim, 50)
        self.l2 = nn.Linear(50, x_dim + 2)
        self.height = height
        self.x_dim = x_dim
        self.dist_func = dist_func
        initial_weights = torch.ones(1, 1, 1)
        self.register_buffer('initial_weights', initial_weights)
        self.leaf_builder = leaf_builder
        self.temperature = temperature

    def set_dist_func(self, dist_func):
        self.dist_func = dist_func

    def get_decision_params(self, representation):
        params = self.l1(representation)
        params = nn.functional.relu(params)
        params = self.l2(params)
        weights = params[:, :self.x_dim]
        biases = params[:, self.x_dim: -1]
        betas = params[:, -1:]
        return weights, biases, betas

    def get_node(self, r, representations, x, weights, height):
        local_representation = weighted_mean(representations, weights)
        representation = torch.cat((local_representation, r), dim=1)
        if height == 0:
            return self.leaf_builder.get_node(representation)

        node_weights, node_biases, node_betas = self.get_decision_params(representation)
        fake_node = Tree(node_weights, node_biases, node_betas, None, None, self.dist_func, self.temperature)

        right_probs = fake_node.get_right_probs(x).detach()
        right_weights_mask = right_probs.round()
        right_weights = weights * right_weights_mask
        left_weights = weights * (1 - right_weights_mask)
        right_node = self.get_node(r, representations, x, right_weights, height - 1)
        left_node = self.get_node(r, representations, x, left_weights, height - 1)

        return Tree(node_weights, node_biases, node_betas, right_node, left_node, self.dist_func, self.temperature)

    def forward(self, representations, x):
        batch_size = x.shape[0]
        n_samples = x.shape[1]
        weights = self.initial_weights.repeat(batch_size, n_samples, 1)
        r = representations.mean(1)
        return self.get_node(r, representations, x, weights, self.height)


class TreeModel(nn.Module):
    def __init__(self, x_dim, r_dim, y_dim, height, leaf_builder, dist_func, temperature):
        super(TreeModel, self).__init__()
        self.encoder = Encoder(x_dim, r_dim, y_dim)
        self.decoder = TreeBuilder(height, x_dim, r_dim, leaf_builder, dist_func, temperature)
        self.hard = False

    def set_dist_func(self, dist_func):
        self.decoder.set_dist_func(dist_func)

    def make_hard_routing(self):
        self.hard = True

    def get_tree(self, x_train, y_train):
        representations = self.encoder(x_train, y_train)
        return self.decoder(representations, x_train)

    def forward(self, x_train, y_train, x_test):
        tree = self.get_tree(x_train, y_train)
        if self.hard:
            return tree.predict_hard(x_test)
        return tree.predict(x_test)


def get_regression_tree(x_dim, min_y, max_y, r_dim, height, dist_func, temperature):
    leaf_builder = LeafBuilderRegression(r_dim, min_y, max_y)
    model = TreeModel(x_dim, r_dim, 1, height, leaf_builder, dist_func, temperature)
    return model


def get_classification_tree(x_dim, y_dim, r_dim, height, dist_func, temperature):
    leaf_builder = LeafBuilderClassification(r_dim, y_dim)
    model = TreeModel(x_dim, r_dim, y_dim, height, leaf_builder, dist_func, temperature)
    return model
