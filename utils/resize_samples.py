if __name__ == '__main__':
    import argparse

    from load_data import get_samples_filenames
    from image_prep import bulk_process_samples, load_resize_store

    parser = argparse.ArgumentParser(description='Resize image samples.')
    parser.add_argument('-s', '--source', type=str, help='Source directory of the samples')
    parser.add_argument('-o', '--output', type=str, help='Output directory of the samples')
    parser.add_argument('-r', '--res', type=tuple, help='Final image dimensions', default=(256, 256))
    parser.add_argument('-sf', '--suffix', type=int, help='Resized sample filename suffix', default=None)

    args = parser.parse_args()

    samples = get_samples_filenames(args.source)
    bulk_process_samples(args.source, args.output, samples, load_resize_store, args.res, args.suffix)
