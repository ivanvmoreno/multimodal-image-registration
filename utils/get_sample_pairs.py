if __name__ == '__main__':
    import argparse

    from load_data import get_samples_filenames, get_sample_pairs

    parser = argparse.ArgumentParser(description='Process pyramidal TIF samples.')
    parser.add_argument('-s', '--source', type=str, help='Source directory of the samples')
    parser.add_argument('-o', '--output', type=str, help='Output directory of the txt containing the pairs')
    parser.add_argument('-t', '--tag', type=str, help='Tag to be used to identify H&E samples (e.g. "HE")', default='HE')

    args = parser.parse_args()

    samples = get_samples_filenames(args.source)
    pairs = get_sample_pairs(samples, args.tag)

    with open(args.output, 'w') as f:
        for p in pairs:
            f.write(f'{p[0]},{p[1]}\n')
        f.flush()
