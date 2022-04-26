import wave
from pydub import AudioSegment

audio = AudioSegment.from_wav("Fata Morgana Podcast Matija.wav")

#cut_1_seconds = 60 * 46 + 58
n_frames = 3694054
#idx_cut_1 = int(cut_1_seconds / audio.duration_seconds * n_frames)
#audio_cut_1 = audio[:idx_cut_1]
#cut_2_seconds = 60 * 48 + 43
#idx_cut_2 = int(cut_2_seconds / audio.duration_seconds * n_frames)
#audio_cut_2 = audio[idx_cut_2:]
#audio_new = audio_cut_1.append(audio_cut_2)
#audio_new.export("mix_cut.wav")
audio_new = audio

highlight1_start_seconds = 60 * 5 + 45
highlight1_end_seconds = 60 * 6 + 0
idx_highlight1_start = int(highlight1_start_seconds / audio.duration_seconds * n_frames)
idx_highlight1_end = int(highlight1_end_seconds / audio.duration_seconds * n_frames)
high1 = audio_new[idx_highlight1_start:idx_highlight1_end]
high1.export("high1.wav")
highlight2_start_seconds = 60 * 11 + 20
highlight2_end_seconds = 60 * 11 + 36
idx_highlight2_start = int(highlight2_start_seconds / audio.duration_seconds * n_frames)
idx_highlight2_end = int(highlight2_end_seconds / audio.duration_seconds * n_frames)
high2 = audio_new[idx_highlight2_start:idx_highlight2_end]
high2.export("high2.wav")
highlight3_start_seconds = 60 * 36 + 28
highlight3_end_seconds = 60 * 36 + 43
idx_highlight3_start = int(highlight3_start_seconds / audio.duration_seconds * n_frames)
idx_highlight3_end = int(highlight3_end_seconds / audio.duration_seconds * n_frames)
high3 = audio_new[idx_highlight3_start:idx_highlight3_end]
high3.export("high3.wav")
highlight4_start_seconds = 60 * 37 + 1
highlight4_end_seconds = 60 * 37 + 16
idx_highlight4_start = int(highlight4_start_seconds / audio.duration_seconds * n_frames)
idx_highlight4_end = int(highlight4_end_seconds / audio.duration_seconds * n_frames)
high4 = audio_new[idx_highlight4_start:idx_highlight4_end]
high4.export("high4.wav")
highlight5_start_seconds = 60 * 47 + 10
highlight5_end_seconds = 60 * 47 + 25
idx_highlight5_start = int(highlight5_start_seconds / audio.duration_seconds * n_frames)
idx_highlight5_end = int(highlight5_end_seconds / audio.duration_seconds * n_frames)
high5 = audio_new[idx_highlight5_start:idx_highlight5_end]
high5.export("high5.wav")



