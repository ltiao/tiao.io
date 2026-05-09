import sys
import click

import json
import requests
import requests_cache

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import mplfinance as mpf
import seaborn as sns

from tqdm import trange
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import chain
from pathlib import Path


GOLDEN_RATIO = 0.5 * (1 + np.sqrt(5))
WIDTH = 397.48499


def pt_to_in(x):
    pt_per_in = 72.27
    return x / pt_per_in


def size(width, aspect=GOLDEN_RATIO):
    width_in = pt_to_in(width)
    return (width_in, width_in / aspect)


def create_trades_frame(trades):

    frame = pd.DataFrame(trades).assign(
        time=lambda trade: pd.to_datetime(trade.time, unit="ms"),
        price=lambda trade: pd.to_numeric(trade.price),
        qty=lambda trade: pd.to_numeric(trade.qty),
        quoteQty=lambda trade: pd.to_numeric(trade.quoteQty),
    )
    return frame


@click.command()
@click.argument(
    "output_dir", default="figures/", type=click.Path(file_okay=False, dir_okay=True)
)
@click.option("--num-samples", "-s", default=5)
@click.option("--transparent", is_flag=True)
@click.option("--context", default="paper")
@click.option("--style", default="ticks")
@click.option("--palette", default="muted")
@click.option("--palette-minor", default="crest")
@click.option("--width", "-w", type=float, default=pt_to_in(WIDTH))
@click.option("--height", "-h", type=float)
@click.option("--aspect", "-a", type=float, default=GOLDEN_RATIO)
@click.option("--dpi", type=float, default=300)
@click.option("--extension", "-e", multiple=True, default=["png"])
def main(
    output_dir,
    num_samples,
    transparent,
    context,
    style,
    palette,
    palette_minor,
    width,
    height,
    aspect,
    dpi,
    extension,
):

    # preamble
    if height is None:
        height = width / aspect
    # height *= num_iterations
    # figsize = size(width, aspect)
    figsize = (width, height)

    suffix = f"{width*dpi:.0f}x{height*dpi:.0f}"

    rc = {
        "figure.figsize": figsize,
        "font.serif": ["Times New Roman"],
        "text.usetex": True,
    }
    sns.set(context=context, style=style, palette=palette, font="serif", rc=rc)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # constants
    limit = 500
    # num_blocks_list = [100, 100]
    num_blocks = 100

    API_KEY = "LRw4YGARO2RwF5PdkGh4mfQdCiuofY4C9qYB79cyMCiYTKAcmFwZmzwFuXqfCqYK"
    headers = {"X-MBX-APIKEY": API_KEY}

    s = requests_cache.CachedSession(backend="sqlite", expire_after=6000)
    # s = requests.Session()
    s.headers.update(headers)

    target = "ETH"
    source = "AUD"
    symbol = "".join((target, source))

    r = s.get("https://api.binance.com/api/v3/ticker/price", params=dict(symbol=symbol))
    print(json.dumps(r.json(), sort_keys=True, indent=2))

    r = s.get("https://api.binance.com/api/v3/aggTrades", params=dict(symbol=symbol))
    print(json.dumps(r.json(), sort_keys=True, indent=2))
    frame = pd.DataFrame(data=r.json(), dtype="float64").assign(
        T=lambda row: pd.to_datetime(row["T"], unit="ms")
    )

    fig, ax = plt.subplots()

    sns.scatterplot(x="T", y="p", marker="+", data=frame, ax=ax)  # s=0.5, alpha=0.2,

    sns.despine(fig=fig, ax=ax, offset=1, trim=True)
    fig.autofmt_xdate()

    plt.tight_layout()

    for ext in extension:
        fig.savefig(
            output_path.joinpath(f"aggTrades_{context}_{suffix}.{ext}"),
            dpi=dpi,
            transparent=transparent,
        )

    plt.show()

    return 0

    r = s.get("https://api.binance.com/api/v3/avgPrice", params=dict(symbol=symbol))
    print(json.dumps(r.json(), sort_keys=True, indent=2))

    r = s.get(
        "https://api.binance.com/api/v3/klines",
        params=dict(symbol=symbol, interval="4h"),
    )
    print(json.dumps(r.json(), sort_keys=True, indent=2))

    frame = (
        pd.DataFrame(
            data=r.json(),
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ],
            dtype="float64",
        )
        .assign(
            open_time=lambda interval: pd.to_datetime(interval.open_time, unit="ms"),
            close_time=lambda interval: pd.to_datetime(interval.close_time, unit="ms"),
        )
        .set_index("close_time")
    )

    fig, (ax1, ax2) = plt.subplots(nrows=2, gridspec_kw=dict(height_ratios=[3, 1]))

    mpf.plot(frame, type="candle", style="binance", volume=ax2, ax=ax1)

    sns.despine(fig=fig, ax=ax1, offset=1, trim=True)
    fig.autofmt_xdate()

    plt.tight_layout()

    for ext in extension:
        fig.savefig(
            output_path.joinpath(f"kline_{context}_{suffix}.{ext}"),
            dpi=dpi,
            transparent=transparent,
        )

    plt.show()

    ###

    url = "https://api.binance.com/api/v3/trades"
    params = dict(symbol=symbol, limit=limit)

    r = s.get(url, params=params)
    block = r.json()

    data = create_trades_frame(block).set_index("time")

    print(json.dumps(block, sort_keys=True, indent=2))
    print(data.head().to_markdown())

    fig, ax = plt.subplots()

    sns.scatterplot(
        x="time",
        y="price",
        marker="+",  # s=0.5, alpha=0.2,
        data=data.reset_index(),
        ax=ax,
    )

    sns.despine(fig=fig, ax=ax, offset=1, trim=True)
    fig.autofmt_xdate()

    plt.tight_layout()

    for ext in extension:
        fig.savefig(
            output_path.joinpath(f"recent_{context}_{suffix}.{ext}"),
            dpi=dpi,
            transparent=transparent,
        )

    plt.show()

    ###

    blocks = []
    blocks.append(block)
    for i in trange(num_blocks):
        params["fromId"] = block[0]["id"] - limit  # get first trade of previous block
        r = s.get("https://api.binance.com/api/v3/historicalTrades", params=params)
        block = r.json()
        blocks.append(block)

    data = create_trades_frame(chain(*blocks))

    fig, ax = plt.subplots()

    sns.scatterplot(x="time", y="price", marker="x", s=0.5, alpha=0.2, data=data, ax=ax)

    sns.despine(fig=fig, ax=ax, offset=1, trim=True)
    fig.autofmt_xdate()

    plt.tight_layout()

    for ext in extension:
        fig.savefig(
            output_path.joinpath(f"price_{context}_{suffix}.{ext}"),
            dpi=dpi,
            transparent=transparent,
        )

    plt.show()

    data_ohlc = data.set_index("time").price.resample("30T").ohlc()

    print(data_ohlc.head().to_markdown())

    fig, ax = plt.subplots()

    # sns.scatterplot(x="time", y="open", marker="x", data=df, ax=ax)
    # sns.scatterplot(x="time", y="close", marker="x", data=df, ax=ax)
    # sns.scatterplot(x="time", y="high", marker="+", data=df, ax=ax)
    # sns.scatterplot(x="time", y="low", marker="+", data=df, ax=ax)

    mpf.plot(data_ohlc, type="candle", style="binance", ax=ax)

    sns.despine(fig=fig, ax=ax, offset=1, trim=True)
    fig.autofmt_xdate()

    plt.tight_layout()

    for ext in extension:
        fig.savefig(
            output_path.joinpath(f"candlestick_{context}_{suffix}.{ext}"),
            dpi=dpi,
            transparent=transparent,
        )

    plt.show()

    data_volume = data.set_index("time").quoteQty.resample("30T").sum()
    frame = data_ohlc.assign(volume=data_volume)

    print(frame.head().to_markdown())

    fig, (ax1, ax2) = plt.subplots(nrows=2, gridspec_kw=dict(height_ratios=[3, 1]))

    # sns.scatterplot(x="time", y="open", marker="x", data=df, ax=ax)
    # sns.scatterplot(x="time", y="close", marker="x", data=df, ax=ax)
    # sns.scatterplot(x="time", y="high", marker="+", data=df, ax=ax)
    # sns.scatterplot(x="time", y="low", marker="+", data=df, ax=ax)

    # divider = make_axes_locatable(ax)
    # ax_volume = divider.append_axes("bottom", size=0.6, pad=0.1, sharex=ax1)

    mpf.plot(frame, type="candle", style="binance", volume=ax2, ax=ax1)

    sns.despine(fig=fig, ax=ax1, offset=1, trim=True)
    fig.autofmt_xdate()

    plt.tight_layout()

    for ext in extension:
        fig.savefig(
            output_path.joinpath(f"combined_{context}_{suffix}.{ext}"),
            dpi=dpi,
            transparent=transparent,
        )

    plt.show()

    # open_ = data["price"].resample("30T", origin="start").first()
    # close = data["price"].resample("30T", origin="start", label="right").last()

    # low = data["price"].resample("30T", origin="start").min()
    # # low.index = low.index + to_offset("2.5T")

    # high = data["price"].resample("30T", origin="start").max()
    # # high.index = high.index + to_offset("2.5T")

    # fig, ax = plt.subplots()

    # sns.scatterplot(x="time", y="price", marker="x", s=0.5, data=data, alpha=0.2, ax=ax)
    # sns.scatterplot(x="time", y="price", marker="x", data=open_.reset_index(), ax=ax)
    # sns.scatterplot(x="time", y="price", marker="x", data=close.reset_index(), ax=ax)
    # sns.scatterplot(x="time", y="price", marker="+", data=low.reset_index(), ax=ax)
    # sns.scatterplot(x="time", y="price", marker="+", data=high.reset_index(), ax=ax)

    # xs = open_.index.to_pydatetime()
    # ys = open_.to_numpy()

    # us = close.index.to_pydatetime()
    # vs = close.to_numpy()

    # # ax.scatter(xs, ys)

    # for (x, y, u, v, l, h) in zip(xs, ys, us, vs, low.to_numpy(), high.to_numpy()):
    #     color = "tab:green" if v > y else "tab:red"
    #     rect = Rectangle((x, y), u - x, v - y, linewidth=1, edgecolor=color, facecolor="none")
    #     ax.add_patch(rect)

    #     ax.hlines(l, x, u, colors=color)
    #     ax.hlines(h, x, u, colors=color)

    # sns.despine(fig=fig, ax=ax, offset=1, trim=True)
    # fig.autofmt_xdate()

    # plt.tight_layout()

    # for ext in extension:
    #     fig.savefig(output_path.joinpath(f"price_{context}_{suffix}.{ext}"),
    #                 dpi=dpi, transparent=transparent)

    # plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
