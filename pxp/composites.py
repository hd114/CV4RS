from zennit.composites import EpsilonPlusFlat, ZPlus, Epsilon
from zennit.rules import Alpha1Beta0IgnoreBn

def get_cnn_composite(model_name, rule_map=None):
    """
    Create a composite for CNN models based on predefined rules.

    Args:
        model_name (str): Name of the CNN model (e.g., "resnet50").
        rule_map (dict): Custom mapping of layer types to rules.

    Returns:
        zennit.composites.EpsilonPlusFlat: Composite with configured rules for relevance propagation.
    """
    if rule_map is None:
        rule_map = {
            "low_level_hidden_layer_rule": "Epsilon",
            "mid_level_hidden_layer_rule": "EpsilonPlusFlat",
            "high_level_hidden_layer_rule": "ZPlus",
            "fully_connected_layers_rule": "Epsilon",
            "softmax_rule": "Epsilon",
        }

    composite = EpsilonPlusFlat()

    # Assign rules based on the model's layer architecture
    for rule_name, rule_type in rule_map.items():
        if rule_type == "Epsilon":
            composite.add_rule(Alpha1Beta0IgnoreBn(Epsilon()))
        elif rule_type == "EpsilonPlusFlat":
            composite.add_rule(EpsilonPlusFlat())
        elif rule_type == "ZPlus":
            composite.add_rule(ZPlus())

    print(f"Composite for {model_name} initialized with rules: {rule_map}")
    return composite
