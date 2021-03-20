import json
import subprocess
import os


def download_cut(video):
    exists = []
    for e in video[1].items():
        exists.append(os.path.exists(
            'finegym/videos/%s/%s.mp4' % (video[0], e[0])))
    if all(exists):
        print('Already cut %s.' % video[0])
        return
    if os.path.exists('/tmp/%s.mp4' % video[0]):
        print('Already downloaded %s.' % video[0])
    else:
        command = 'youtube-dl --quiet --no-warnings --no-check-certificate -f mp4 -o "/tmp/%s.mp4" "https://www.youtube.com/watch?v=%s"' % (
            video[0], video[0])
        print(command)
        attempts = 0
        while True:
            try:
                subprocess.check_output(
                    command, shell=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                attempts += 1
                if attempts == 4:
                    print('Can\'t download %s.' % video[0])
                return
            else:
                print('Downloaded %s.' % video[0])
                break
    os.makedirs('finegym/videos/%s' % video[0], exist_ok=True)
    command = 'ffmpeg -i "/tmp/%s.mp4"' % video[0]
    for e in video[1].items():
        command += ' -ss %f -t %f finegym/videos/%s/%s.mp4' % (e[1]['timestamps']
                                                               [0][0], e[1]['timestamps'][0][1]-e[1]['timestamps'][0][0], video[0], e[0])
    command += ' -c:v libx264 -c:a copy -threads 12 -loglevel panic'
    print(command)
    try:
        subprocess.check_output(
            command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        print('Can\'t cut %s.' % video[0])
    else:
        print('Cut %s.' % video[0])
    os.remove('/tmp/%s.mp4' % video[0])


videos = json.load(open('data/finegym_annotation_info_v1.1.json'))
for video in videos.items():
    download_cut(video)
