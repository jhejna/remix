from openx.data.datasets import oxe
from openx.data.datasets.bridge import bridge_dataset_transform
from openx.utils.spec import ModuleSpec

OXE_ALL = dict(
    rt1=dict(
        path="fractal20220817_data/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        transform=ModuleSpec.create(oxe.rt1_dataset_transform),
        weight=3513684,
    ),
    kuka=dict(
        path="kuka/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        transform=ModuleSpec.create(oxe.kuka_dataset_transform),
        weight=2133779,
    ),
    bridge=dict(
        path="bridge_dataset/1.0.0",
        train_split="train",
        val_split="val",
        transform=ModuleSpec.create(bridge_dataset_transform),
        weight=1946218,
    ),
    taco_play=dict(
        path="taco_play/0.1.0",
        train_split="train",
        val_split="test",
        transform=ModuleSpec.create(oxe.taco_dataset_transform),
        weight=210730,
    ),
    taco_extra=dict(
        path="taco_extra/1.0.0",
        train_split="train[:95%]",
        val_split=None,  # Ignore this one since we have val data in taco_play
        transform=ModuleSpec.create(oxe.taco_dataset_transform),
        weight=51756,
    ),
    jaco_play=dict(
        path="jaco_play/0.1.0",
        train_split="train",
        val_split="test",
        transform=ModuleSpec.create(oxe.jaco_play_dataset_transform),
        weight=69151,
    ),
    berkeley_cable_routing=dict(
        path="berkeley_cable_routing/0.1.0",
        train_split="train",
        val_split="test",
        transform=ModuleSpec.create(oxe.berkeley_cable_routing_dataset_transform),
        weight=36758,
    ),
    roboturk=dict(
        path="roboturk/0.1.0",
        train_split="train",
        val_split="test",
        transform=ModuleSpec.create(oxe.roboturk_dataset_transform),
        weight=166627,
    ),
    viola=dict(
        path="viola/0.1.0",
        train_split="train",
        val_split="test",
        transform=ModuleSpec.create(oxe.viola_dataset_transform),
        weight=68778,
    ),
    berkeley_autolab_ur5=dict(
        path="berkeley_autolab_ur5/0.1.0",
        train_split="train",
        val_split="test",
        transform=ModuleSpec.create(oxe.berkeley_autolab_ur5_dataset_transform),
        weight=86887,
    ),
    toto=dict(
        path="toto/0.1.0",
        train_split="train",
        val_split="test",
        transform=ModuleSpec.create(oxe.toto_dataset_transform),
        weight=293237,
    ),
    language_table=dict(
        path="language_table/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        transform=ModuleSpec.create(oxe.language_table_dataset_transform),
        weight=6275438,
    ),
    stanford_hydra=dict(
        path="stanford_hydra_dataset_converted_externally_to_rlds/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        transform=ModuleSpec.create(oxe.stanford_hydra_dataset_transform),
        weight=339237,
    ),
    austin_buds=dict(
        path="austin_buds_dataset_converted_externally_to_rlds/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        transform=ModuleSpec.create(oxe.austin_buds_dataset_transform),
        weight=32757,
    ),
    furniture_bench=dict(
        path="furniture_bench_dataset_converted_externally_to_rlds/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        transform=ModuleSpec.create(oxe.furniture_bench_dataset_transform),
        weight=3739948,
    ),
    ucsd_kitchen=dict(
        path="ucsd_kitchen_dataset_converted_externally_to_rlds/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        transform=ModuleSpec.create(oxe.ucsd_kitchen_dataset_transform),
        weight=3614,
    ),
    austin_sailor=dict(
        path="austin_sailor_dataset_converted_externally_to_rlds/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        transform=ModuleSpec.create(oxe.austin_sailor_dataset_transform),
        weight=334011,
    ),
    austin_sirius=dict(
        path="austin_sirius_dataset_converted_externally_to_rlds/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        transform=ModuleSpec.create(oxe.austin_sirius_dataset_transform),
        weight=265812,
    ),
    bc_z=dict(
        path="bc_z/1.0.0",
        train_split="train",
        val_split="eval",
        transform=ModuleSpec.create(oxe.bc_z_dataset_transform),
        weight=5432343,
    ),
    dlr_shared_control=dict(
        path="dlr_edan_shared_control_converted_externally_to_rlds/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        transform=ModuleSpec.create(oxe.dlr_edan_shared_control_dataset_transform),
        weight=8487,
    ),
    iamlab_cmu_pickup_insert=dict(
        path="iamlab_cmu_pickup_insert_converted_externally_to_rlds/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        transform=ModuleSpec.create(oxe.iamlab_pick_insert_dataset_transform),
        weight=138937,
    ),
    utaustin_mutex=dict(
        path="utaustin_mutex_resize/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        transform=ModuleSpec.create(oxe.utaustin_mutex_dataset_transform),
        weight=343702,
    ),
    berkeley_fanuc_manipulation=dict(
        path="berkeley_fanuc_manipulation_resize/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        transform=ModuleSpec.create(oxe.berkeley_fanuc_dataset_transform),
        weight=58781,
    ),
    cmu_stretch=dict(
        path="cmu_stretch_resize/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        transform=ModuleSpec.create(oxe.cmu_stretch_dataset_transform),
        weight=23556,
    ),
    #### the below datasets are NOT in the Octo Magic Soup Mix
    nyu_franka_play=dict(
        path="nyu_franka_play_dataset_converted_externally_to_rlds/0.1.0",
        train_split="train",
        val_split="val",
        transform=ModuleSpec.create(oxe.nyu_franka_play_dataset_transform),
        weight=34083,
    ),
    maniskill=dict(
        path="maniskill_dataset_converted_externally_to_rlds/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        transform=ModuleSpec.create(oxe.maniskill_dataset_transform),
        weight=4282929,
    ),
    cmu_franka_exploration=dict(
        path="cmu_franka_exploration_dataset_converted_externally_to_rlds/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        transform=ModuleSpec.create(oxe.cmu_franka_exploration_dataset_transform),
        weight=34083,
    ),
    kaist_nonprehensile=dict(
        path="kaist_nonprehensile_converted_externally_to_rlds/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        transform=ModuleSpec.create(oxe.kaist_nonprehensible_dataset_transform),
        weight=30499,
    ),
    stanford_robocook=dict(
        path="stanford_robocook_converted_externally_to_rlds/0.1.0",
        train_split="train[:95%]",
        val_split="train[95%:]",
        transform=ModuleSpec.create(oxe.robocook_dataset_transform),
        weight=104863,
    ),
    # Remove because its currently causing NaNs
    # cmu_playfusion=dict(
    #     path="cmu_play_fusion/0.1.0",
    #     train_split="train[:95%]",
    #     val_split="train[95%:]",
    #     transform=ModuleSpec.create(oxe.playfusion_dataset_transform),
    #     weight=221887,
    # ),
    # Has wierd actions? Seems like EE_POS might be switched with EULER
    # rh20t=dict(
    #     path="rh20t/1.0.0",
    #     train_split="train[:95%]",
    #     val_split="train[95%:]",
    #     transform=ModuleSpec.create(oxe.rh20t_dataset_transform),
    #     weight=3513815,
    # ),
)

OXE_MAGIC_SOUP = {
    dataset: OXE_ALL[dataset] | dict(weight=OXE_ALL[dataset]["weight"] * weight)
    for dataset, weight in [
        ("rt1", 0.54087122203),
        ("kuka", 0.8341046294),
        ("bridge", 1.0),
        ("taco_play", 2.0),
        ("taco_extra", 2.0),
        ("jaco_play", 1.0),
        ("berkeley_cable_routing", 1.0),
        ("roboturk", 2.0),
        # ("nyu_door_opening_surprising_effectiveness", 1.0), Removed since we aren't training wrist.
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 2.0),
        ("toto", 1.0),
        ("language_table", 0.1),
        ("stanford_hydra", 2.0),
        ("austin_buds", 1.0),
        ("nyu_franka_play", 3.0),
        ("furniture_bench", 0.1),
        ("austin_sailor", 1.0),
        ("austin_sirius", 1.0),
        ("bc_z", 0.2),
        ("dlr_shared_control", 1.0),
        ("iamlab_cmu_pickup_insert", 1.0),
        ("utaustin_mutex", 1.0),
        ("berkeley_fanuc_manipulation", 2.0),
        ("cmu_stretch", 1.0),
    ]
}

# The same as the magic soup mix, except do not apply weightings
OXE_MAGIC_SOUP_UNIFORM = {dataset: OXE_ALL[dataset] for dataset in OXE_MAGIC_SOUP.keys()}


OXE_EXPANDED_SOUP = {
    dataset: OXE_ALL[dataset] | dict(weight=OXE_ALL[dataset]["weight"] * weight)
    for dataset, weight in [
        ("rt1", 0.4),
        ("kuka", 0.8341046294),
        ("bridge", 1.0),
        ("taco_play", 2.0),
        ("taco_extra", 2.0),
        ("jaco_play", 1.0),
        ("berkeley_cable_routing", 1.0),
        ("roboturk", 2.0),
        # ("nyu_door_opening_surprising_effectiveness", 1.0), Removed since we aren't training wrist.
        ("viola", 1.0),
        ("berkeley_autolab_ur5", 2.0),
        ("toto", 1.0),
        ("language_table", 0.1),
        ("stanford_hydra", 2.0),
        ("austin_buds", 1.0),
        ("nyu_franka_play", 3.0),
        ("furniture_bench", 0.1),
        # ("ucsd_kitchen", 2.0), # Removed for Wierd actions.
        ("austin_sailor", 1.0),
        ("austin_sirius", 1.0),
        ("bc_z", 0.2),
        ("dlr_shared_control", 1.0),
        ("iamlab_cmu_pickup_insert", 1.0),
        ("utaustin_mutex", 1.0),
        ("berkeley_fanuc_manipulation", 2.0),
        ("cmu_stretch", 1.0),
        # ("rh20t", 0.1), # Seems to have wierd actions, will investigate later.
    ]
}

# The same as the magic soup mix, except do not apply weightings
OXE_EXPANDED_SOUP_UNIFORM = {dataset: OXE_ALL[dataset] for dataset in OXE_EXPANDED_SOUP.keys()}


RTX_MIX = {
    dataset: OXE_ALL[dataset] | dict(weight=OXE_ALL[dataset]["weight"] * weight)
    for dataset, weight in [
        ("rt1", 0.54087122203),
        ("kuka", 0.8341046294),
        ("bridge", 1.0),
        ("taco_play", 2.0),
        ("taco_extra", 2.0),
        ("jaco_play", 2.0),
        ("berkeley_cable_routing", 3.0),
        ("roboturk", 1.0),
        # ("nyu_door_opening_surprising_effectiveness", 5.0), Removed since we don't have wrist
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 1.0),
        ("toto", 1.0),
    ]
}

RTX_MIX_UNIFORM = {dataset: OXE_ALL[dataset] for dataset in RTX_MIX.keys()}

RTX_DOREMI_150K = {
    dataset: OXE_ALL[dataset] | dict(weight=weight)
    for dataset, weight in [
        ("berkeley_autolab_ur5", 0.02368840016424656),
        ("berkeley_cable_routing", 0.0019966200925409794),
        ("bridge", 0.1991041600704193),
        ("jaco_play", 0.0038572715129703283),
        ("kuka", 0.12065707892179489),
        ("roboturk", 0.011355401016771793),
        ("rt1", 0.3944754898548126),
        ("taco_extra", 0.006291169673204422),
        ("taco_play", 0.03037598729133606),
        ("toto", 0.19304856657981873),
        ("viola", 0.015149127691984177),
    ]
}


RTX_DOREMI_200K = {
    dataset: OXE_ALL[dataset] | dict(weight=weight)
    for dataset, weight in [
        ("berkeley_autolab_ur5", 0.019247833639383316),
        ("berkeley_cable_routing", 0.002363319508731365),
        ("bridge", 0.19114349782466888),
        ("jaco_play", 0.004058686085045338),
        ("kuka", 0.1237783432006836),
        ("roboturk", 0.011293914169073105),
        ("rt1", 0.45421281456947327),
        ("taco_extra", 0.007307005580514669),
        ("taco_play", 0.04522952809929848),
        ("toto", 0.12147394567728043),
        ("viola", 0.01989024691283703),
    ]
}
