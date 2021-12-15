from PIL import Image, ImageDraw, ImageFont
from mpi4py import MPI
import timeit, sys, os

image_path = sys.argv[1]

start = timeit.default_timer()

comm = MPI.COMM_WORLD
processor_name = MPI.Get_processor_name()
rank = comm.Get_rank()
size = comm.Get_size()

def get_concat_h(im1, im2):
    image = Image.new("RGB", (im1.width + im2.width, im1.height))
    image.paste(im1,(0,0))
    image.paste(im2,(im1.width,0))

    return image

def water_mark(image, text, width, height):
    draw = ImageDraw.Draw(image)
    #font = ImageFont.load_default()

    font = ImageFont.truetype(font='Ubuntu-B.ttf', size=80)
    text_width,text_height = draw.textsize(text,font)

    x = width + text_width
    y = height + text_height

    draw.text((x,y), text, font=font)


def filter_blue(r, g, b):
    return (0, 0, b)

def filter_red(r, g, b):
    return (r, 0, 0)

def filter_green(r,g,b):
    return (0, g, 0)

def execute_filter(width_start, height_start,width, height, fun_color, pixels):
    for p_y in range(height_start,height):
        for p_x in range(width_start, width):
            r,g,b = pixels[p_x,p_y]
            pixels[p_x, p_y] = fun_color(r,g,b)

if rank == 0:
    print("PROCESS PARALLEL.\n")

    img = Image.open(image_path).convert("RGB")

    width,height = img.size

    left = width / 2
    top = 0
    right = width
    bottom = height

    part_1 = img.crop((0,0,left, height))
    part_2 = img.crop(((left), top, width, bottom))

    req = comm.send(part_1, dest=1, tag=111)
    req1 = comm.send(part_2, dest=2, tag=112)

elif rank == 1:
    req2 = comm.recv(source=0, tag=111)
    part_1 = req2

    pixels_1 = part_1.load()

    width,height = part_1.size

    print("Processing image part 1 on processor {0}".format(processor_name))

    execute_filter(0, 0,width, height, filter_blue, pixels_1)
    water_mark(part_1, processor_name, 0, 0)

    comm.send(part_1, dest=0, tag=1)

    name_part_1 = 'rank_{0}_v3.jpg'.format(rank)
    part_1.save(name_part_1)

    os.system(f'gpicview {name_part_1}')

elif rank == 2:
    req3 = comm.recv(source=0, tag=112)
    part_2 = req3

    pixels_2 = part_2.load()

    width,height = part_2.size

    print("Processing image part 2 on processor {0}".format(processor_name))

    execute_filter(0, 0,width, height, filter_green, pixels_2)
    water_mark(part_2, processor_name, 0, 0)

    comm.send(part_2, dest=0, tag=2)

    name_part_2 = 'rank_{0}_v3.jpg'.format(rank)
    part_2.save(name_part_2)

    os.system('gpicview {name_part_2}')

if rank == 0:
    req2 = comm.recv(source=1, tag=1)
    req3 = comm.recv(source=2, tag=2)

    part_1 = req2
    part_2 = req3

    print("\nMount final image")

    img_saida = get_concat_h(part_1, part_2)
    name_saida = 'saida_v3.jpg'
    img_saida.save(name_saida)

    stop = timeit.default_timer()
    print("Precess time: {:.2f}".format((stop-start)))

    os.system(f'gpicview {name_saida}')

MPI.Finalize()
