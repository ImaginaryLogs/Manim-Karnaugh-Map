all: build run 
build:
	@echo "Running..."
	python -m manim .\test\main.py KarnaughMap

run:
	@echo "Playing..."
	start ./media/videos/main/1080p60/KarnaughMap.mp4 
