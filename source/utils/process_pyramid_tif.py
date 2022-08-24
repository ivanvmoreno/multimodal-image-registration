if __name__ == '__main__':
    import argparse

    from load_data import get_samples_filenames
    from image_prep import bulk_process_samples, load_store_tif_page

    parser = argparse.ArgumentParser(description='Process pyramidal TIF samples.')
    parser.add_argument('-s', '--source', type=str, help='Source directory')
    parser.add_argument('-o', '--output', type=str, help='Output directory')
    parser.add_argument('-l', '--level', help='TIF level to be loaded', default='lowest')
    parser.add_argument('-sf', '--suffix', type=int, help='Resulting filename suffix', default=None)

    args = parser.parse_args()

    samples = get_samples_filenames(args.source)
    bulk_process_samples(args.source, args.output, samples, load_store_tif_page, args.level, args.suffix)
