import pandas as pd


def calulcate_distance(df):
    print(df['pos_x'])


def summarize(read_path):
    df = pd.read_csv(read_path)

    # Grppup the runs
    groupby_labels = ['iteration', 'exp_id', 'navigation_type']
    grouped = df.groupby(by=groupby_labels)

    # Infractions
    infractions = grouped[
        [
            'acceleration',
            'speed',
            'collision_predistrain',
            'collision_vehicle',
            'collision_other',
            'n_lane_invasion',
        ]
    ].mean()
    print(infractions)

    print(grouped.apply(calulcate_distance))
