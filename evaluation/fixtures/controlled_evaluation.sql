-- Controlled execution-aware evaluation fixture schema.
-- Used by evaluation tests to verify SQL execution correctness.

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    role TEXT NOT NULL,
    status TEXT DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS berths (
    id INTEGER PRIMARY KEY,
    berth_name TEXT NOT NULL,
    berth_code TEXT NOT NULL UNIQUE,
    status TEXT DEFAULT 'available',
    terminal_id INTEGER REFERENCES terminals(id)
);

CREATE TABLE IF NOT EXISTS vessels (
    id INTEGER PRIMARY KEY,
    vessel_name TEXT NOT NULL,
    imo_number TEXT UNIQUE,
    vessel_type TEXT DEFAULT 'cargo'
);

CREATE TABLE IF NOT EXISTS terminals (
    id INTEGER PRIMARY KEY,
    terminal_name TEXT NOT NULL,
    location TEXT
);

CREATE TABLE IF NOT EXISTS service_orders (
    id INTEGER PRIMARY KEY,
    cost NUMERIC NOT NULL,
    status TEXT DEFAULT 'pending',
    user_id INTEGER REFERENCES users(id),
    berth_id INTEGER REFERENCES berths(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS assignments (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    berth_id INTEGER NOT NULL REFERENCES berths(id),
    status TEXT DEFAULT 'active',
    assigned_date DATE DEFAULT CURRENT_DATE
);

CREATE TABLE IF NOT EXISTS vessel_berth_assignments (
    id INTEGER PRIMARY KEY,
    vessel_id INTEGER NOT NULL REFERENCES vessels(id),
    berth_id INTEGER NOT NULL REFERENCES berths(id),
    status TEXT DEFAULT 'active',
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Seed data
INSERT INTO users (id, name, role, status) VALUES
    (1, 'Alice', 'admin', 'active'),
    (2, 'Bob', 'operator', 'active'),
    (3, 'Charlie', 'viewer', 'inactive'),
    (4, 'Diana', 'admin', 'active'),
    (5, 'Eve', 'operator', 'active');

INSERT INTO berths (id, berth_name, berth_code, status, terminal_id) VALUES
    (1, 'Berth Alpha', 'BA01', 'available', 1),
    (2, 'Berth Beta', 'BB02', 'occupied', 2),
    (3, 'Berth Gamma', 'BG03', 'available', 1);

INSERT INTO vessels (id, vessel_name, imo_number, vessel_type) VALUES
    (1, 'MV Navigator', 'IMO1234567', 'cargo'),
    (2, 'SS Explorer', 'IMO7654321', 'tanker');

INSERT INTO terminals (id, terminal_name, location) VALUES
    (1, 'North Terminal', 'Sector A'),
    (2, 'South Terminal', 'Sector B');

INSERT INTO service_orders (id, cost, status, user_id, berth_id) VALUES
    (1, 1500.00, 'completed', 1, 1),
    (2, 2300.50, 'pending', 2, 2),
    (3, 800.00, 'completed', 1, 3),
    (4, 4100.00, 'cancelled', 3, 1),
    (5, 950.75, 'completed', 4, 2);

INSERT INTO assignments (id, user_id, berth_id, status, assigned_date) VALUES
    (1, 1, 1, 'active', '2026-01-15'),
    (2, 2, 2, 'active', '2026-02-01'),
    (3, 3, 3, 'inactive', '2025-12-01'),
    (4, 4, 1, 'active', '2026-03-10');

INSERT INTO vessel_berth_assignments (id, vessel_id, berth_id, status, assigned_at) VALUES
    (1, 1, 1, 'active', '2026-03-01 08:00:00'),
    (2, 2, 2, 'completed', '2026-02-15 10:30:00'),
    (3, 1, 3, 'active', '2026-03-20 06:15:00');
