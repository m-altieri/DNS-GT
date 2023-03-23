import sys
import os
import tensorflow as tf
import numpy as np
import argparse
from models import DELM, Word2Vec
import time
import os
from tqdm.keras import TqdmCallback
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import logging
from colorama import Fore, Style


def config_tf(args):
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    physical_devices = tf.config.list_physical_devices("GPU")
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except:
            print(f"Cannot enable memory growth on some device.")
            sys.exit(1)


def get_logger(verbose=False):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if verbose:
        logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def build_model(model, args):
    if model.lower() == "delm":
        model = DELM(
            seqlen=args.seqlen,
            blocks=args.blocks,
            mask_test=args.mask_test,
            tensorboard=args.tensorboard,
            quick_tb=args.quick_tb,
            run_name=args.run_name,
            omega=args.omega,
            version=args.version,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=False)],
            run_eagerly=args.eager,
        )
    elif model.lower() == "w2v":
        model = Word2Vec(type=args.type)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=False)],
            run_eagerly=args.eager,
        )
    return model


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--es",
        action="store_true",
        help="Early Stopping",
    )
    argparser.add_argument(
        "--load",
        action="store",
        nargs="?",
        const="last",
        help="Whether to load the model from a saved checkpoint or to reinitialize a new one. Set it with no value or with 'last' to load the most recent checkpoint, or manually set a checkpoint file name.",
    )
    argparser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log debug information",
    )
    argparser.add_argument(
        "--epochs",
        action="store",
        default=5,
        type=int,
        help="Number of training epochs",
    )
    argparser.add_argument(
        "--bs", action="store", default=256, type=int, help="Batch size"
    )
    argparser.add_argument(
        "--lr", action="store", default=1e-4, type=float, help="Learning rate"
    )
    argparser.add_argument(
        "--demo",
        action="store_true",
        help="Used for debugging purposes",
    )
    argparser.add_argument(
        "--test-seq",
        action="store",
        type=int,
        help="Used for debugging purposes; choose the test sequence index",
    )
    argparser.add_argument(
        "--seqlen", action="store", default=32, type=int, help="Maximum sequence length"
    )
    argparser.add_argument(
        "--stride",
        action="store",
        default=1,
        type=int,
        help="Stride between sequences (how many queries to shift by)",
    )
    argparser.add_argument(
        "--include-start",
        action="store_true",
        help="Whether to include <START> as the first token of each sequence (total length is unaffected)",
    )
    argparser.add_argument(
        "--version",
        action="store",
        choices=["small", "all"],
        default="small",
        help="Version of the dataset used.",
    )
    argparser.add_argument(
        "--gpu",
        action="store",
        help="Only set it if you are running on a multi-gpu machine (es. --gpu 3)",
    )
    argparser.add_argument("--tensorboard", action="store_true")
    argparser.add_argument(
        "--quick-tb",
        action="store_true",
        help="Whether to reutilize the same TensorBoard folder. Allows for quicker debugging.",
    )
    argparser.add_argument("--eager", action="store_true")
    argparser.add_argument("--blocks", action="store", type=int, default=4)
    argparser.add_argument("--group-hosts", action="store_true", default=True)
    argparser.add_argument(
        "--run-name",
        action="store",
        default=f'model-{time.strftime("%y%m%d-%H%M%S", time.localtime())}.h5',
    )
    argparser.add_argument("--omega", action="store", type=float, default=0.8)
    argparser.add_argument("--shuffle", action="store_true")
    argparser.add_argument("model", action="store", default="DELM")
    argparser.add_argument(
        "--type",
        action="store",
        help="Model type. It is used by model classes that have multiple subtypes, like Word2Vec.",
    )

    args = argparser.parse_args()

    assert args.test_seq is None or args.test_seq > 0

    args.mask_test = not args.demo

    if args.demo:
        args.eager = True
        args.tensorboard = True
    return args


def seq_generator_from_folder(
    input_folder,
    seqlen,
    stride=1,
    include_start=False,
    group_hosts=True,
    model=None,
):
    """Folder containing .npy files, each representing a matrix of shape (n_queries, 2)."""
    for f in os.listdir(input_folder):
        if os.path.splitext(os.path.join(input_folder, f))[-1] != ".npy":
            continue
        seqs = create_sequences(
            os.path.join(input_folder, f),
            seqlen,
            stride,
            include_start,
            group_hosts,
            model,
        )
        for seq in seqs:
            yield seq


def create_sequences(
    input_file,
    seqlen,
    stride=1,
    include_start=False,
    group_hosts=True,
    model=None,
):  # input [queries, 2]

    queries = np.load(input_file, allow_pickle=True)
    if group_hosts:
        queries = queries[
            np.argsort(queries[:, 0])
        ]  # Sort queries by host, preserving row structure

    if model == "delm":  # output [queries - stride, seqlen, 2]
        actual_seqlen = seqlen - include_start
        seqs = np.empty(
            shape=((len(queries) - actual_seqlen) // stride + 1, seqlen, 2),
            dtype=object,
        )
        for i, _ in enumerate(seqs):
            if include_start:
                seqs[i][0] = ["<START>", "<START>"]
            seqs[i][include_start:] = queries[i * stride : i * stride + actual_seqlen]

    elif model == "w2v":  # output [queries, seqlen]
        seqs = np.array(Word2Vec.create_pairs(queries[:, 1], seqlen))

    else:
        raise ValueError("Specify model to create sequences.")

    return seqs


def find_last_checkpoint(dir="checkpoints"):
    if len(os.listdir(dir)) > 0:
        checkpoint = os.listdir(dir)[
            [os.path.getmtime(os.path.join(dir, f)) for f in os.listdir(dir)].index(
                max([os.path.getmtime(os.path.join(dir, f)) for f in os.listdir(dir)])
            )
        ]
    else:
        checkpoint = ""
    return checkpoint


def indent(depth=1):
    return f"".join(["--" for i in range(depth - 1)]) + "> "


def default_checkpoint(args):
    return args.run_name


def main():
    args = parse_args()

    logger = get_logger(args.verbose)
    logger.info("Started training with args:")
    logger.info("\n".join([f"{indent(1)}{k}: {vars(args)[k]}" for k in vars(args)]))

    queries_path = f"preprocessing/arrays/{args.version}/queries/"
    domains_vocab_path = f"preprocessing/vocabs/{args.version}/domains_vocab.txt"
    hosts_vocab_path = f"preprocessing/vocabs/{args.version}/hosts_vocab.txt"

    config_tf(args)

    train = tf.data.Dataset.from_generator(
        lambda: seq_generator_from_folder(
            os.path.join(queries_path, "train"),
            stride=args.stride,
            seqlen=args.seqlen,
            include_start=args.include_start,
            group_hosts=args.group_hosts,
            model=args.model.lower(),
        ),
        output_signature=tf.TensorSpec(shape=(args.seqlen, 2), dtype=tf.string)
        if args.model.lower() == "delm"
        else tf.TensorSpec(shape=(args.seqlen,), dtype=tf.string),
    )
    test = tf.data.Dataset.from_generator(
        lambda: seq_generator_from_folder(
            os.path.join(queries_path, "test"),
            stride=args.stride,
            seqlen=args.seqlen,
            include_start=args.include_start,
            group_hosts=args.group_hosts,
            model=args.model.lower(),
        ),
        output_signature=tf.TensorSpec(shape=(args.seqlen, 2), dtype=tf.string)
        if args.model.lower() == "delm"
        else tf.TensorSpec(shape=(args.seqlen,), dtype=tf.string),
    )

    if not args.demo and args.shuffle:
        train = train.shuffle(1000000)
    train = train.batch(args.bs).prefetch(tf.data.AUTOTUNE)
    test = test.batch(args.bs).prefetch(tf.data.AUTOTUNE)

    model = build_model(args.model, args)

    # Manage checkpoint
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    checkpoint_folder = os.path.join(
        "checkpoints", f"{args.model}{f'-{args.type}' if args.type else ''}"
    )
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    checkpoint_name = default_checkpoint(args)

    if args.load:
        checkpoint_name = (
            find_last_checkpoint(dir=checkpoint_folder)
            if args.load == "last"
            else args.load
        )
        logger.debug(
            f"Trying to load model weights from {os.path.join(checkpoint_folder, checkpoint_name)}..."
        )

        logger.info(f"Calling model to initialize layers...")
        # model(list(train.take(1).unbatch().as_numpy_iterator())[0:1])

        sample = np.array(list(train.take(1).unbatch().as_numpy_iterator())[0:1])
        logger.info(sample)
        model.test_step(sample)

        try:
            model.load_weights(os.path.join(checkpoint_folder, checkpoint_name))
            logger.info(
                f"Model weights loaded from {os.path.join(checkpoint_folder, checkpoint_name)}."
            )
        except Exception as e:
            logger.error(
                f"{Fore.RED}Exception when trying to load checkpoint {checkpoint_name}:\n{Style.DIM}{e}\n{Style.NORMAL}Continuing without loading checkpoint.{Style.RESET_ALL}"
            )
            checkpoint_name = default_checkpoint(args)

    if args.demo:
        logger.info(
            f"{Style.BRIGHT}\nDomain Embeddings Language Model v0.1{Style.RESET_ALL}\n"
            + "Please refer to https://gitlab.jrc.ec.europa.eu/jrc-projects/createg/cdp-bari/dns/-/tree/main/ for roadmap and updates.\n"
            + "Syntax: <Host> <Domain> -> <Predicted Domain> (<prob%>) [(<Unmasked Domain> <prob%>)]\n"
        )

        with open(domains_vocab_path, "r") as f:
            domains_vocab = [l.strip() for l in f.readlines()]

        seq = (
            train.unbatch()
            .skip(args.test_seq or np.random.randint(0, 1000))
            .take(10)
            .as_numpy_iterator()
        )
        seq = np.array([s for s in seq], dtype=object)
        print(seq)
        # uncomment this assignment to manually create a sequence
        # note that arbitrarily created sequences will be harder to predict, since they don't follow any pattern in the training data
        # seq = np.array(
        #     [
        #         [
        #             ["172.31.1.6", "graph.facebook.com"],
        #             ["172.31.1.6", "graph.facebook.com"],
        #             ["172.31.1.6", "graph.facebook.com"],
        #             ["172.31.1.6", "graph.facebook.com"],
        #             ["172.31.1.6", "graph.facebook.com"],
        #             ["172.31.1.6", "graph.facebook.com"],
        #             ["172.31.1.6", "graph.facebook.com"],
        #             ["172.31.1.6", "graph.facebook.com"],
        #             ["172.31.1.6", "graph.facebook.com"],
        #             ["172.31.1.6", "graph.facebook.com"],
        #         ]
        #     ],
        #     dtype=object,
        # )

        mask = np.zeros_like(seq)
        # place 1's where you want to replace tokens with <MASK>
        # axis 0 is always 0 (array of length 1), axis 1 is the index of token within the sequence, axis 2 is 0 for host and 1 for domain
        # example: mask[0, 1, 1]
        #   always zero ^  ^  ^
        #      first token |  |
        #                     | domain

        masked_seq = np.where(mask, np.full_like(seq, "<MASK>", dtype=object), seq)

        pred, loss = model._predict(seq, mask)
        pred = pred[0]

        for i in range(len(pred)):

            masked_host = masked_seq[0, i, 0]
            if type(masked_host) is bytes:
                masked_host = masked_host.decode("utf-8")
            domain = seq[0, i, 1]
            if type(domain) is bytes:
                domain = domain.decode("utf-8")
            masked_domain = masked_seq[0, i, 1]
            if type(masked_domain) is bytes:
                masked_domain = masked_domain.decode("utf-8")

            predicted_token = domains_vocab[np.array(pred).argmax(axis=-1)[i]]
            domain_index = domains_vocab.index(domain)
            logger.info(
                f"{masked_host} {masked_domain} -> {f'{Fore.GREEN}' if domain == predicted_token else f'{Fore.RED}'}{predicted_token} ({100*(np.array(pred).max(axis=-1)[i]):.2f}%){Style.RESET_ALL} {f'{Style.DIM}({domain} {100*(np.array(pred)[i,domain_index]):.2f}%) {Style.RESET_ALL}' if not domain == predicted_token else ''}"
            )

        logger.info(f"{Style.BRIGHT}Loss: {loss:.3f}{Style.RESET_ALL}")
        logger.info(model.summary())

        sys.exit(0)

    logger.debug("Starting model training...")
    model.fit(
        x=train,
        y=None,
        validation_data=test,
        validation_freq=1,
        batch_size=args.bs,
        epochs=args.epochs,
        callbacks=[
            ModelCheckpoint(
                os.path.join(checkpoint_folder, checkpoint_name),
                monitor="loss",
                save_weights_only=True,
            ),
            TensorBoard(
                log_dir=os.path.join("tensorboard", args.run_name),
                histogram_freq=1,
                profile_batch="500,520",
            ),
        ],
    )
    logger.debug(f"Model training completed.")

    model.save_weights(
        os.path.join(checkpoint_folder, checkpoint_name)
    )  # Save model weights

    # Save embeddings
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    embeddings_folder = os.path.join(
        "embeddings", f"{args.model}{f'-{args.type}' if args.type else ''}"
    )
    if not os.path.exists(embeddings_folder):
        os.makedirs(embeddings_folder)

    domain_embeddings = (
        model.domain_embeddings.embeddings.numpy()
    )  # TODO may break if model class uses a different variable name; use a get_embeddings() function instead
    np.save(
        os.path.join(
            embeddings_folder, f"embeddings-{os.path.splitext(checkpoint_name)[0]}.npy"
        ),
        domain_embeddings,
    )

    logger.debug("Starting model evaluation...")
    model.evaluate(x=test, y=None, batch_size=args.bs)
    logger.debug("Model evaluation completed.")


if __name__ == "__main__":
    main()
