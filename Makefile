.PHONY: help install run test clean docker-build docker-run deploy-heroku deploy-docker

help:
	@echo "🎧 Speech Separation System"
	@echo "==========================="
	@echo "make install        - Install dependencies"
	@echo "make run            - Run locally"
	@echo "make test           - Run tests"
	@echo "make docker-build   - Build Docker image"
	@echo "make docker-run     - Run with Docker"
	@echo "make deploy-heroku  - Deploy to Heroku"
	@echo "make clean          - Clean up"

install:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

run:
	streamlit run app_assemblyai.py

test:
	pytest tests/ -v

docker-build:
	docker-compose build

docker-run:
	docker-compose up -d
	docker-compose logs -f

docker-stop:
	docker-compose down

deploy-heroku:
	@bash heroku_deploy.sh

deploy-docker:
	@bash docker_deploy.sh

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf logs/* outputs/* cache/*