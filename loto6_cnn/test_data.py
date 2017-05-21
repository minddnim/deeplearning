import cairo
import io
import numpy as np
import PIL.Image

import random
random.seed(a=0)

def curve_yx(ctx, t):
    xs, ys = ctx.get_current_point()
    xt, yt = t
    ctx.curve_to(xs,(ys+yt)/2,(xs+xt)/2,yt,xt,yt)

def curve_xy(ctx, t):
    xs, ys = ctx.get_current_point()
    xt, yt = t
    ctx.curve_to((xt+xs)/2,ys,xt,(ys+yt)/2,xt,yt)

def arc_y(ctx, t, offset):
    xs, ys = ctx.get_current_point()
    xt, yt = t
    y = max(ys, yt) if offset > 0 else min(ys,yt)
    ctx.curve_to(xs, y + offset, xt, y + offset, xt, yt)

def scurve_y(ctx, t, offset):
    xs, ys = ctx.get_current_point()
    xt, yt = t
    ctx.curve_to(xs, ys+offset, xt, yt-offset, xt, yt)

def scurve_x(ctx, t, offset):
    xs, ys = ctx.get_current_point()
    xt, yt = t
    ctx.curve_to(xs+offset, ys, xt-offset, yt, xt, yt)

def create_outer_path(ctx,
                 left_outer_bottom = (150, 350),
                 left_outer_top = (270, 200),
                 left_outer_center = (320, 220),
                 left_outer_middle = (320, 280),
                 left_tip = (250, 300),
                 left_inner_middle = (280, 280),
                 left_inner_top = (260, 260),
                 left_inner_bottom = (200,330),

                 right_outer_bottom = (490, 350),
                 right_outer_center = (320, 220),
                 right_outer_middle = (320, 280),
                 right_outer_top = (370, 200),
                 right_tip = (330, 330),
                 right_inner_middle = (350,280),
                 right_inner_top = (400, 260),
                 right_inner_bottom = (440,330)
                 ):

    # outer
    ctx.move_to(*left_outer_bottom)
    curve_yx(ctx, left_outer_top)

    curve_xy(ctx, left_outer_center)

    # from outer
    ctx.line_to(*left_outer_middle)

    if left_tip[0] > left_inner_middle[0]:
        curve_yx(ctx, left_tip)
    else:
        if left_tip[1] < left_outer_middle[1] + 20:
            arc_y(ctx, left_tip, 20)
        else:
            curve_yx(ctx, left_tip)

    if left_tip[0] < left_inner_middle[0]:
        scurve_y(ctx, left_inner_middle, -20)
    else:
        curve_xy(ctx, left_inner_middle)

    curve_yx(ctx, left_inner_top)
    curve_xy(ctx, left_inner_bottom)

    arc_y(ctx, right_inner_bottom, 20)

    # inner right
    curve_yx(ctx, right_inner_top)
    curve_xy(ctx, right_inner_middle)

    if right_tip[0] < right_inner_middle[0]:
        curve_yx(ctx, right_tip)
    else:
        scurve_y(ctx, right_tip, 20)

    if right_tip[0] < right_inner_middle[0]:
        curve_xy(ctx, right_outer_middle)
    else:
        if right_outer_middle[1] + 20 > right_tip[1]:
            arc_y(ctx, right_outer_middle, 20)
        else:
            curve_xy(ctx, right_outer_middle)

    ctx.line_to(*right_outer_center)

    curve_yx(ctx, right_outer_top)
    curve_xy(ctx, right_outer_bottom)

    arc_y(ctx, left_outer_bottom, 30)

    ctx.close_path()

def create_outer_outer_path(ctx, **d):
    ctx.move_to(*d['left_outer_bottom'])
    curve_yx(ctx, d['left_outer_top'])

    curve_xy(ctx, d['left_outer_center'])
    ctx.line_to(*d['right_outer_center'])
    curve_yx(ctx, d['right_outer_top'])
    curve_xy(ctx, d['right_outer_bottom'])

    arc_y(ctx, d['left_outer_bottom'], 30)

    ctx.close_path()

def random_pos(x_range, y_range):
    return (random.randrange(*x_range), random.randrange(*y_range))

def create_outer_param():
    d = {}
    d['left_outer_bottom'] = random_pos((140, 160), (340, 360))
    d['left_outer_top'] = random_pos((200,290), (100, 200))
    d['left_outer_center'] = (320, d['left_outer_top'][1] + random.randrange(10, 50))

    d['left_inner_bottom'] = (d['left_outer_bottom'][0]+50, d['left_outer_bottom'][1]-20)
    d['left_inner_top'] = (random.randrange(min(d['left_outer_top'][0] - 10, d['left_outer_top'][0]+10), 310), d['left_outer_top'][1] + 60)

    d['left_inner_middle'] = (random.randrange(d['left_inner_top'][0], 315), d['left_inner_top'][1] + random.randrange(0,20))
    d['left_outer_middle'] = (320, max(d['left_outer_center'][1], d['left_inner_middle'][1]) + random.randrange(0,30))

    left_tip_x = random.randrange(d['left_inner_bottom'][0], 320)
    if left_tip_x > d['left_inner_middle'][0]:
        left_tip_y = random.randrange(max(d['left_inner_middle'][1], d['left_outer_middle'][1]), d['left_inner_bottom'][1]+10)
    else:
        left_tip_y = random.randrange(min(d['left_inner_middle'][1], d['left_outer_middle'][1]), d['left_inner_bottom'][1]+10)
    d['left_tip'] = (left_tip_x, left_tip_y)

    d['right_outer_bottom'] = random_pos((480, 500), (340, 360))
    d['right_outer_top'] = (random.randrange(350, 440), d['left_outer_center'][1] + random.randrange(-40, -5))
    d['right_outer_center'] = d['left_outer_center']

    d['right_inner_bottom'] = (d['right_outer_bottom'][0]-50, d['right_outer_bottom'][1]-20)
    d['right_inner_top'] = (random.randrange(340, max(d['right_outer_top'][0] + 10, d['right_outer_top'][0]-10)), d['right_outer_top'][1] + 60)

    d['right_inner_middle'] = (random.randrange(330, d['right_inner_top'][0]), d['right_inner_top'][1] + random.randrange(0,20))
    d['right_outer_middle'] = (320, max(d['right_outer_center'][1], d['right_inner_middle'][1]) + random.randrange(0,30))

    right_tip_x = random.randrange(320, d['right_inner_bottom'][0])
    if right_tip_x < d['right_inner_middle'][0]:
        right_tip_y = random.randrange(max(d['right_inner_middle'][1], d['right_outer_middle'][1]), d['right_inner_bottom'][1]+10)
    else:
        right_tip_y = random.randrange(min(d['right_inner_middle'][1], d['right_outer_middle'][1]), d['right_inner_bottom'][1]+10)
    d['right_tip'] = (right_tip_x, right_tip_y)

    return d

def create_inner_path(ctx,
                      left_bottom = (200, 330),
                      left_top = (260, 260),
                      center = (320, 280),
                      right_top = (400, 260),
                      right_bottom = (440, 330)):
    ctx.move_to(*left_bottom)
    curve_yx(ctx, left_top)
    scurve_x(ctx, center, 20)
    scurve_x(ctx, right_top, 20)
    curve_xy(ctx, right_bottom)
    arc_y(ctx, left_bottom, 20)
    ctx.close_path()

def create_inner_param(outer_param):
    def rand_range(p, offset):
        return (p[0] + random.randrange(-offset, offset), p[1] + random.randrange(-offset, offset))

    d = {}
    d['left_bottom'] = rand_range(outer_param['left_inner_bottom'], 10)
    d['left_top'] = rand_range(outer_param['left_inner_top'], 10)
    d['center'] = (320, min(outer_param['left_outer_middle'][1], outer_param['right_outer_middle'][1]))
    d['right_top'] = rand_range(outer_param['right_inner_top'], 10)
    d['right_bottom'] = rand_range(outer_param['right_inner_bottom'], 10)
    return d

def surface_to_pil(surface):
    buf = io.BytesIO()
    surface.write_to_png(buf)
    return PIL.Image.open(buf)

def create_train_image(outer_param, inner_param):
    surface=cairo.ImageSurface(cairo.FORMAT_ARGB32, 640, 480)
    ctx=cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_NONE)

    ctx.set_source_rgb(0, 0, 0)
    ctx.rectangle(0,0,surface.get_width(), surface.get_height())
    ctx.fill()

    create_inner_path(ctx, **inner_param)
    ctx.set_source_rgb(0, 1, 0)
    ctx.fill_preserve()
    ctx.set_source_rgb(0, 0, 1)
    ctx.stroke()

    create_outer_path(ctx, **outer_param)
    ctx.set_source_rgb(1, 0, 0)
    ctx.fill_preserve()
    ctx.set_source_rgb(0,0,1)
    ctx.stroke()

    return surface_to_pil(surface)

def create_input_image(outer_param, inner_param):
    surface=cairo.ImageSurface(cairo.FORMAT_ARGB32, 640, 480)
    ctx=cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_NONE)

    ctx.set_source_rgb(0, 0, 0)
    ctx.rectangle(0,0,surface.get_width(), surface.get_height())
    ctx.fill()

    create_inner_path(ctx, **inner_param)
    ctx.set_source_rgb(0.8, 0.8, 0.8)
    ctx.fill()

    create_outer_path(ctx, **outer_param)
    ctx.set_source_rgb(0.8,0.8,0.8)
    ctx.fill_preserve()
    ctx.set_source_rgb(0.6,0.6,0.6)
    ctx.set_dash([14,10, 5, 5])
    ctx.stroke()

    ctx.set_source_rgb(0,0,0)
    ctx.set_dash([])
    create_outer_outer_path(ctx, **outer_param)
    ctx.stroke()

    return surface_to_pil(surface)

def image_to_labels(image):
    src = np.asarray(image)
    ary = np.zeros(src.shape[:2], dtype=np.int32)
    m = src[:,:,2] == 255
    ary[m] = -1
    m = src[:,:,0] == 255
    ary[m] = 0
    m = src[:,:,1] == 255
    ary[m] = 1
    m = (src == [0,0,0]).all(2)
    ary[m] = 2
    return ary

def create_data(mini_batch_size):
    in_channels = 1
    in_size = (480, 640)

    xs = np.zeros((mini_batch_size, in_channels,  *in_size)).astype(np.float32)
    ys = np.zeros((mini_batch_size, *in_size)).astype(np.int32)

    for i in range(mini_batch_size):
        outer_param = create_outer_param()
        inner_param = create_inner_param(outer_param)

        in_image = create_input_image(outer_param, inner_param)
        in_data = in_image / np.linalg.norm(in_image)
        xs[i, 0, :, :] = np.dot(in_data[...,:3], [0.299, 0.587, 0.114])

        train_image = create_train_image(outer_param, inner_param)
        ys[i, :, :] = image_to_labels(train_image)

    return xs, ys

def main():
    [[xs]], [ys] = create_data(1)

    import matplotlib.pyplot as plt

    plt.subplot(211)
    plt.imshow(xs, cmap=plt.get_cmap('gray'))

    plt.subplot(212)
    plt.imshow(ys)
    plt.show()

if __name__ == '__main__':
    main()
