from typing import Any
import scripts.untitled.operators as opr

MERGEMODES_LIST = []
CALCMODES_LIST = []

# ============================================================================
# BASE CLASSES
# ============================================================================

class MergeMode:
    """Defines the STRUCTURE of the merge formula (what models combine and how)"""
    name = 'mergemode'
    description = 'description'
    input_models = 4
    input_sliders = 3

    slid_a_info = '-'
    slid_a_config = (-1, 2, 0.01) #minimum,maximum,step

    slid_b_info = '-'
    slid_b_config = (-1, 2, 0.01)

    slid_c_info = '-'
    slid_c_config = (-1, 2, 0.01)

    slid_d_info = '-'
    slid_d_config = (-1, 2, 0.01)

    def create_recipe(self, key, model_a, model_b, model_c, model_d, seed=False, alpha=0, beta=0, gamma=0, delta=0) -> opr.Operation:
        """Create the base operation tree structure"""
        raise NotImplementedError


class CalcMode:
    """Defines HOW operations are calculated (modifies merge mode execution)"""
    name = 'calcmode'
    description = 'description'
    compatible_modes = ['all']  # Which merge modes this works with ('all' or list of mode names)

    def modify_recipe(self, recipe, key, model_a, model_b, model_c, model_d, **kwargs) -> opr.Operation:
        """Modify the recipe created by a MergeMode. Default: return unchanged"""
        return recipe


# ============================================================================
# MERGE MODES (Formula Structures)
# ============================================================================

class WeightSum(MergeMode):
    name = 'Weight-Sum'
    description = 'model_a * (1 - alpha) + model_b * alpha'
    input_models = 2
    input_sliders = 1
    slid_a_info = "model_a - model_b"
    slid_a_config = (0, 1, 0.01)

    def create_recipe(key, model_a, model_b, model_c, model_d, alpha=0, **kwargs):
        a = opr.LoadTensor(key,model_a)
        b = opr.LoadTensor(key,model_b)

        if alpha >= 1:
            return b
        elif alpha <= 0:
            return a

        c = opr.Multiply(key, 1-alpha, a)
        d = opr.Multiply(key, alpha, b)

        res = opr.Add(key, c, d)
        return res

MERGEMODES_LIST.append(WeightSum)


class AddDifference(MergeMode):
    name = 'Add Difference'
    description = 'model_a + (model_b - model_c) * alpha'
    input_models = 3
    input_sliders = 1
    slid_a_info = "addition multiplier"
    slid_a_config = (-1, 2, 0.01)
    slid_b_info = "smooth (slow)"
    slid_b_config = (0, 1, 1)

    def create_recipe(key, model_a, model_b, model_c, model_d, alpha=0, beta=0, **kwargs):
        a = opr.LoadTensor(key,model_a)
        b = opr.LoadTensor(key,model_b)
        c = opr.LoadTensor(key,model_c)

        # This will be potentially replaced by CalcModes like TrainDifference
        diff = opr.Sub(key, b, c)
        if beta == 1:
            diff = opr.Smooth(key,diff)
        diff.cache()

        diffm = opr.Multiply(key, alpha, diff)

        res = opr.Add(key, a, diffm)
        return res

MERGEMODES_LIST.append(AddDifference)


class TripleSum(MergeMode):
    name = 'Triple Sum'
    description = 'model_a * alpha + model_b * beta + model_c * gamma'
    input_models = 3
    input_sliders = 3
    slid_a_info = "model_a weight"
    slid_a_config = (0, 1, 0.01)
    slid_b_info = "model_b weight"
    slid_b_config = (0, 1, 0.01)
    slid_c_info = "model_c weight"
    slid_c_config = (0, 1, 0.01)

    def create_recipe(key, model_a, model_b, model_c, model_d, alpha=0.33, beta=0.33, gamma=0.34, **kwargs):
        a = opr.LoadTensor(key,model_a)
        b = opr.LoadTensor(key,model_b)
        c = opr.LoadTensor(key,model_c)

        a_weighted = opr.Multiply(key, alpha, a)
        b_weighted = opr.Multiply(key, beta, b)
        c_weighted = opr.Multiply(key, gamma, c)

        ab = opr.Add(key, a_weighted, b_weighted)
        res = opr.Add(key, ab, c_weighted)
        return res

MERGEMODES_LIST.append(TripleSum)


class SumTwice(MergeMode):
    name = 'Sum Twice'
    description = '(1-beta)*((1-alpha)*model_a + alpha*model_b) + beta*model_c'
    input_models = 3
    input_sliders = 2
    slid_a_info = "model_a - model_b"
    slid_a_config = (0, 1, 0.01)
    slid_b_info = "result - model_c"
    slid_b_config = (0, 1, 0.01)

    def create_recipe(key, model_a, model_b, model_c, model_d, alpha=0.5, beta=0.5, **kwargs):
        a = opr.LoadTensor(key,model_a)
        b = opr.LoadTensor(key,model_b)
        c = opr.LoadTensor(key,model_c)

        # First merge: (1-alpha)*A + alpha*B
        a_weighted = opr.Multiply(key, 1-alpha, a)
        b_weighted = opr.Multiply(key, alpha, b)
        first_merge = opr.Add(key, a_weighted, b_weighted)

        # Second merge: (1-beta)*first_merge + beta*C
        first_weighted = opr.Multiply(key, 1-beta, first_merge)
        c_weighted = opr.Multiply(key, beta, c)

        res = opr.Add(key, first_weighted, c_weighted)
        return res

MERGEMODES_LIST.append(SumTwice)


class QuadSum(MergeMode):
    name = 'Quad Sum'
    description = 'model_a * alpha + model_b * beta + model_c * gamma + model_d * delta (EXPERIMENTAL - may produce artifacts)'
    input_models = 4
    input_sliders = 4
    slid_a_info = "model_a weight"
    slid_a_config = (0, 1, 0.01)
    slid_b_info = "model_b weight"
    slid_b_config = (0, 1, 0.01)
    slid_c_info = "model_c weight"
    slid_c_config = (0, 1, 0.01)
    slid_d_info = "model_d weight"
    slid_d_config = (0, 1, 0.01)

    def create_recipe(key, model_a, model_b, model_c, model_d, alpha=0.25, beta=0.25, gamma=0.25, delta=0.25, **kwargs):
        a = opr.LoadTensor(key,model_a)
        b = opr.LoadTensor(key,model_b)
        c = opr.LoadTensor(key,model_c)
        d = opr.LoadTensor(key,model_d)

        a_weighted = opr.Multiply(key, alpha, a)
        b_weighted = opr.Multiply(key, beta, b)
        c_weighted = opr.Multiply(key, gamma, c)
        d_weighted = opr.Multiply(key, delta, d)

        ab = opr.Add(key, a_weighted, b_weighted)
        abc = opr.Add(key, ab, c_weighted)
        res = opr.Add(key, abc, d_weighted)
        return res

MERGEMODES_LIST.append(QuadSum)


# ============================================================================
# CALCULATION MODES (How Operations Are Computed)
# ============================================================================

class Normal(CalcMode):
    """Standard arithmetic - no modifications"""
    name = 'normal'
    description = 'Standard calculation (no modifications)'
    compatible_modes = ['all']

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, **kwargs):
        return recipe

CALCMODES_LIST.append(Normal)


class TrainDifferenceCalc(CalcMode):
    """Replaces difference operations with adaptive TrainDiff"""
    name = 'trainDifference'
    description = 'Treats difference as fine-tuning with adaptive scaling'
    compatible_modes = ['Add Difference', 'Triple Sum', 'Sum Twice']  # Works with modes that have differences

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0, beta=0, **kwargs):
        # Replace Sub operation with TrainDiff in AddDifference recipes
        a = opr.LoadTensor(key,model_a)
        b = opr.LoadTensor(key,model_b)
        c = opr.LoadTensor(key,model_c)

        diff = opr.TrainDiff(key,a, b, c)
        diff.cache()

        diffm = opr.Multiply(key, alpha, diff)
        res = opr.Add(key, a, diffm)
        return res

CALCMODES_LIST.append(TrainDifferenceCalc)


class ExtractCalc(CalcMode):
    """Similarity-based feature extraction"""
    name = 'extract'
    description = 'Adds (dis)similar features between models using cosine similarity'
    compatible_modes = ['Add Difference']

    slid_a_info = 'model_b - model_c'
    slid_a_config = (0, 1, 0.01)
    slid_b_info = 'similarity - dissimilarity'
    slid_b_config = (0, 1, 0.01)
    slid_c_info = 'similarity bias'
    slid_c_config = (0, 2, 0.01)
    slid_d_info = 'addition multiplier'
    slid_d_config = (-1, 4, 0.01)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0, beta=0, gamma=0, delta=1, **kwargs):
        a = opr.LoadTensor(key,model_a)
        b = opr.LoadTensor(key,model_b)
        c = opr.LoadTensor(key,model_c)

        extracted = opr.Extract(key, alpha, beta, gamma*15, a, b, c)
        extracted.cache()

        multiplied = opr.Multiply(key, delta, extracted)
        res = opr.Add(key, a, multiplied)
        return res

CALCMODES_LIST.append(ExtractCalc)


class TensorCalc(CalcMode):
    """Exchanges entire tensors by probability"""
    name = 'tensor'
    description = 'Swaps entire tensors from A or B based on probability (not weighted blend)'
    compatible_modes = ['Weight-Sum']

    slid_a_info = "probability of using model_b"
    slid_a_config = (0, 1, 0.01)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0.5, seed=0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        b = opr.LoadTensor(key, model_b)
        res = opr.TensorExchange(key, alpha, seed, a, b)
        return res

CALCMODES_LIST.append(TensorCalc)


class SelfCalc(CalcMode):
    """Multiply model weights by scalar"""
    name = 'self'
    description = 'Multiply model weights by scalar value (single model operation)'
    compatible_modes = ['Weight-Sum']  # Works as modification of WeightSum with one model

    slid_a_info = "weight multiplier"
    slid_a_config = (0, 2, 0.01)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=1.0, **kwargs):
        a = opr.LoadTensor(key, model_a)
        res = opr.Multiply(key, alpha, a)
        return res

CALCMODES_LIST.append(SelfCalc)


class InterpDifferenceCalc(CalcMode):
    """Comparative interpolation based on value differences"""
    name = 'Comparative Interp'
    description = 'Interpolates between values depending on their difference relative to other values'
    compatible_modes = ['Weight-Sum']

    slid_a_info = "concave - convex"
    slid_a_config = (0, 1, 0.01)
    slid_b_info = "similarity - difference"
    slid_b_config = (0, 1, 1)
    slid_c_info = "binomial - linear"
    slid_c_config = (0, 1, 0.01)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0, beta=0, gamma=0, seed=0, **kwargs):
        a = opr.LoadTensor(key,model_a)
        # Skip embeddings
        if key.startswith('cond_stage_model.transformer.text_model.embeddings') or key.startswith('conditioner.embedders.0.transformer.text_model.embeddings') or key.startswith('conditioner.embedders.1.model.token_embedding') or key.startswith('conditioner.embedders.1.model.positional_embedding'):
            return a
        b = opr.LoadTensor(key,model_b)
        return opr.InterpolateDifference(key, alpha, beta, gamma, seed, a ,b)

CALCMODES_LIST.append(InterpDifferenceCalc)


class ManEnhInterpDifferenceCalc(CalcMode):
    """Enhanced interpolation with manual threshold control"""
    name = 'Enhanced Man Interp'
    description = 'Enhanced interpolation with manual threshold control'
    compatible_modes = ['Weight-Sum']

    slid_a_info = "interpolation strength"
    slid_a_config = (0, 1, 0.001)
    slid_b_info = "lower mean threshold"
    slid_b_config = (0, 1, 0.001)
    slid_c_info = "upper mean threshold"
    slid_c_config = (0, 1, 0.001)
    slid_d_info = "smoothness factor"
    slid_d_config = (0, 1, 0.001)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0, beta=0, gamma=0, delta=0, seed=0, **kwargs):
        a = opr.LoadTensor(key,model_a)
        if key.startswith('cond_stage_model.transformer.text_model.embeddings'):
            return a
        b = opr.LoadTensor(key,model_b)
        return opr.ManualEnhancedInterpolateDifference(key, alpha, beta, gamma, delta, seed, a ,b)

CALCMODES_LIST.append(ManEnhInterpDifferenceCalc)


class AutoEnhInterpDifferenceCalc(CalcMode):
    """Enhanced interpolation with automatic threshold calculation"""
    name = 'Enhanced Auto Interp'
    description = 'Interpolates with automatic threshold calculation'
    compatible_modes = ['Weight-Sum']

    slid_a_info = "interpolation strength"
    slid_a_config = (0, 1, 0.001)
    slid_b_info = "threshold adjustment factor"
    slid_b_config = (0, 1, 0.001)
    slid_c_info = "smoothness factor"
    slid_c_config = (0, 1, 0.001)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0, beta=0, gamma=0, seed=0, **kwargs):
        a = opr.LoadTensor(key,model_a)
        if key.startswith('cond_stage_model.transformer.text_model.embeddings'):
            return a
        b = opr.LoadTensor(key,model_b)
        return opr.AutoEnhancedInterpolateDifference(key, alpha, beta, gamma, seed, a ,b)

CALCMODES_LIST.append(AutoEnhInterpDifferenceCalc)


class PowerUpCalc(CalcMode):
    """DARE - Drop And REscale for merging capabilities"""
    name = 'Power-up (DARE)'
    description = 'Adds the capabilities of model B to model A using dropout and rescaling'
    compatible_modes = ['Weight-Sum', 'Add Difference']

    slid_a_info = "dropout rate"
    slid_a_config = (0, 1, 0.01)
    slid_b_info = "addition multiplier"
    slid_b_config = (-1, 4, 0.01)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0, beta=0, seed=0, **kwargs):
        a = opr.LoadTensor(key,model_a)
        b = opr.LoadTensor(key,model_b)

        deltahat = opr.PowerUp(key, alpha, seed, a, b)
        deltahat.cache()

        res = opr.Multiply(key, beta, deltahat)
        return opr.Add(key, a, res)

CALCMODES_LIST.append(PowerUpCalc)


class AddDissimilarityCalc(CalcMode):
    """Add dissimilar features between models"""
    name = 'Add Dissimilarities'
    description = 'Adds dissimilar features between model_b and model_c to model_a'
    compatible_modes = ['Add Difference']

    slid_a_info = 'model_b - model_c'
    slid_a_config = (0, 1, 0.01)
    slid_b_info = 'addition multiplier'
    slid_b_config = (-1, 4, 0.01)
    slid_c_info = 'similarity bias'
    slid_c_config = (0, 2, 0.01)

    def modify_recipe(recipe, key, model_a, model_b, model_c, model_d, alpha=0, beta=0, gamma=0, **kwargs):
        a = opr.LoadTensor(key,model_a)
        b = opr.LoadTensor(key,model_b)
        c = opr.LoadTensor(key,model_c)

        extracted = opr.Similarities(key, alpha, 1, gamma*15, b, c)
        extracted.cache()

        multiplied = opr.Multiply(key, beta, extracted)
        res = opr.Add(key, a, multiplied)
        return res

CALCMODES_LIST.append(AddDissimilarityCalc)
