# copy supervisor config file to the supervisor's default configuration location
cp chatbot_engine.conf /etc/supervisor/conf.d/
echo "Copied chatbot_engine.conf to supervisor's location."

# update the supervisor configs
supervisorctl update
echo "Updated chatbot_engine services."
