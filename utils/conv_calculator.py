def conv_output_calc(h, w, k=(0, 0), p=(0, 0), d=(1, 1), s=(1, 1)):
    if isinstance(k, int):
        k = (k, k)
    if isinstance(p, int):
        p = (p, p)
    if isinstance(d, int):
        d = (d, d)
    if isinstance(s, int):
        s = (s, s)

    h_out = ((h + (2 * p[0]) - d[0] * (k[0] - 1) - 1) / s[0]) + 1
    w_out = ((w + (2 * p[1]) - d[1] * (k[1] - 1) - 1) / s[1]) + 1
    return int(h_out), int(w_out)


def tranconv_output_calc(h, w, k=(0, 0), p=(0, 0), d=(1, 1), s=(1, 1), out_p=(0, 0)):
    if isinstance(k, int):
        k = (k, k)
    if isinstance(p, int):
        p = (p, p)
    if isinstance(d, int):
        d = (d, d)
    if isinstance(s, int):
        s = (s, s)
    if isinstance(out_p, int):
        out_p = (out_p, out_p)

    h_out = (h - 1) * s[0] - 2 * p[0] + d[0] * (k[0] - 1) + out_p[0] + 1
    w_out = (w - 1) * s[1] - 2 * p[1] + d[1] * (k[1] - 1) + out_p[1] + 1
    return int(h_out), int(w_out)


if __name__ == "__main__":
    # print(conv_output_calc(7, 7, k=3, s=5, p=0))
    print(tranconv_output_calc(7, 7, k=3, s=2, p=0))