#!/bin/bash
set -e

# Function to create database and user
create_db_user() {
    local database=$1
    local user=$2
    local password=$3

    echo "Creating database '$database' with user '$user'"

    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
        CREATE USER $user WITH PASSWORD '$password';
        CREATE DATABASE $database OWNER $user;
        GRANT ALL PRIVILEGES ON DATABASE $database TO $user;
EOSQL
}

# Parse POSTGRES_MULTIPLE_DATABASES
if [ -n "$POSTGRES_MULTIPLE_DATABASES" ]; then
    echo "Creating multiple databases..."
    for db_config in $(echo $POSTGRES_MULTIPLE_DATABASES | tr ',' ' '); do
        IFS=':' read -r database user password <<< "$db_config"
        create_db_user "$database" "$user" "$password"
    done
fi

echo "PostgreSQL initialization complete!"
