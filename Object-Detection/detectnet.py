
import argparse
import sys

from jetson_inference import detectNet
from jetson_utils import Log, videoOutput, videoSource

parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.",
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="",
                    nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="",
                    nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2",
                    help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf",
                    help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5,
                    help="minimum detection threshold to use")

try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)


input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)


net = detectNet(args.network, sys.argv, args.threshold)


while True:

    img = input.Capture()

    if img is None:
        continue

    detections = net.Detect(img, overlay=args.overlay)

    print("detected {:d} objects in image".format(len(detections)))
    with open('detected_objects.txt', 'a') as f:
        for detection in detections:
            f.write(f"{detections}\n")

    for detection in detections:

        print(detection.Confidence)

    output.Render(img)

    output.SetStatus("{:s} | Network {:.0f} FPS".format(
        args.network, net.GetNetworkFPS()))

    net.PrintProfilerTimes()

    if not input.IsStreaming() or not output.IsStreaming():
        break
