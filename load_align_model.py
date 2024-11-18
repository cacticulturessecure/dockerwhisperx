from whisperx.alignment import load_align_model

def load_model(language):
    load_align_model(language)

if __name__ == "__main__":
    import sys
    load_model(sys.argv[1])
