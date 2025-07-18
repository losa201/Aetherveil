# Aetherveil Sentinel Knowledge Graph

A comprehensive knowledge graph implementation for cybersecurity intelligence, providing advanced graph analytics, attack path analysis, and threat intelligence correlation.

## Overview

The Knowledge Graph serves as the central intelligence hub for the Aetherveil Sentinel system, tracking all discovered assets, vulnerabilities, threats, and their relationships. It provides sophisticated analytics capabilities for security analysis, pattern recognition, and risk assessment.

## Architecture

### Core Components

1. **GraphManager** - Central hub for graph operations with Neo4j and NetworkX support
2. **GraphSchema** - Comprehensive schema definition and validation
3. **AttackPathAnalyzer** - Advanced attack path analysis and shortest path algorithms
4. **VulnerabilityMapper** - Vulnerability mapping and threat attribution
5. **GraphAlgorithms** - Community detection, centrality measures, and clustering
6. **GraphVisualizer** - Graph visualization and layout generation
7. **GraphAnalytics** - Advanced analytics and intelligence correlation
8. **GraphMaintenance** - Backup, restore, optimization, and health monitoring

### Database Support

- **Primary**: Neo4j graph database with full ACID compliance
- **Fallback**: NetworkX in-memory graphs with persistence
- **Caching**: Redis for performance optimization
- **Hybrid**: Automatic fallback when Neo4j is unavailable

## Features

### 🔍 Graph Analytics
- **Pattern Recognition**: Detect lateral movement, privilege escalation, data exfiltration patterns
- **Intelligence Correlation**: Temporal, technical, and behavioral correlation analysis
- **Trend Analysis**: Historical trend analysis with anomaly detection
- **Risk Scoring**: Comprehensive multi-factor risk assessment

### 🛡️ Security Analysis
- **Attack Path Discovery**: Find optimal attack paths between entities
- **Attack Surface Analysis**: Comprehensive attack surface mapping
- **Vulnerability Clustering**: Identify vulnerability patterns and clusters
- **Threat Attribution**: Correlate threats with known actors and campaigns

### 📊 Visualization
- **Multiple Layouts**: Force-directed, hierarchical, circular, grid layouts
- **Interactive Exploration**: Node expansion, filtering, and search
- **Path Visualization**: Attack path highlighting and context
- **Network Mapping**: Network topology visualization

### 🔧 Operations
- **Backup & Restore**: Full and incremental backup with compression
- **Health Monitoring**: Comprehensive health checks and diagnostics
- **Performance Optimization**: Graph cleanup, indexing, and defragmentation
- **Maintenance Scheduling**: Automated maintenance task scheduling

## Installation

### Dependencies

```bash
# Core graph dependencies
pip install neo4j==5.14.1
pip install networkx==3.2.1
pip install redis==5.0.1

# Analytics dependencies
pip install scikit-learn==1.3.2
pip install scipy==1.11.4
pip install numpy==1.25.2

# Community detection
pip install python-louvain==0.16
pip install leidenalg==0.10.1
pip install igraph==0.11.3

# Visualization
pip install matplotlib==3.8.2
pip install pillow==10.1.0
```

### Configuration

```python
# config/config.py
class DatabaseConfig:
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = "password"
    neo4j_pool_size: int = 50
    neo4j_timeout: int = 30
    
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[SecretStr] = None
```

## Usage

### Basic Graph Operations

```python
from knowledge_graph import GraphManager, GraphSchema
from knowledge_graph.graph_schema import NodeType, RelationType

# Initialize graph manager
graph_manager = GraphManager()
await graph_manager.initialize()

# Create nodes
host_node = await graph_manager.create_node(
    node_type=NodeType.HOST,
    properties={
        "ip": "192.168.1.100",
        "hostname": "web-server-01",
        "os": "linux"
    }
)

vulnerability_node = await graph_manager.create_node(
    node_type=NodeType.VULNERABILITY,
    properties={
        "cve_id": "CVE-2023-1234",
        "severity": "critical",
        "description": "Remote code execution vulnerability"
    }
)

# Create relationships
await graph_manager.create_edge(
    source_id=vulnerability_node.id,
    target_id=host_node.id,
    edge_type=RelationType.AFFECTS,
    properties={"exploitable": True}
)
```

### Attack Path Analysis

```python
from knowledge_graph import AttackPathAnalyzer

analyzer = AttackPathAnalyzer(graph_manager)

# Find attack paths
attack_paths = await analyzer.find_shortest_attack_paths(
    source_node_id="attacker_node",
    target_node_id="target_host",
    max_paths=5
)

# Analyze attack surface
attack_surface = await analyzer.analyze_attack_surface(
    target_node_id="critical_server",
    max_distance=3
)
```

### Advanced Analytics

```python
from knowledge_graph import GraphAnalytics

analytics = GraphAnalytics(graph_manager)

# Detect security patterns
patterns = await analytics.analyze_patterns([
    "lateral_movement",
    "privilege_escalation",
    "data_exfiltration"
])

# Correlate intelligence
correlations = await analytics.correlate_intelligence(
    time_window_hours=24
)

# Calculate risk scores
risk_scores = await analytics.calculate_comprehensive_risk_scores()
```

### Graph Algorithms

```python
from knowledge_graph import GraphAlgorithms

algorithms = GraphAlgorithms(graph_manager)

# Community detection
communities = await algorithms.detect_communities(
    algorithm="louvain",
    resolution=1.0
)

# Centrality analysis
centrality_measures = await algorithms.calculate_centrality_measures()

# Anomaly detection
anomalies = await algorithms.detect_anomalies(
    method="isolation_forest"
)
```

### Visualization

```python
from knowledge_graph import GraphVisualizer

visualizer = GraphVisualizer(graph_manager)

# Generate force-directed layout
layout = await visualizer.generate_layout(GraphLayoutQuery(
    layout_algorithm="force_directed",
    max_nodes=100
))

# Visualize attack path
path_viz = await visualizer.visualize_path(PathVisualizationQuery(
    path_nodes=["node1", "node2", "node3"],
    highlight_critical=True
))
```

### Maintenance Operations

```python
from knowledge_graph import GraphMaintenance

maintenance = GraphMaintenance(graph_manager)

# Create backup
backup_info = await maintenance.create_backup(
    backup_type="full",
    compress=True
)

# Health check
health_checks = await maintenance.perform_health_check()

# Optimize graph
optimization_results = await maintenance.optimize_graph([
    "cleanup", "defragment", "rebuild_indexes"
])
```

## API Endpoints

### Graph Operations
- `POST /api/graph/nodes` - Create node
- `GET /api/graph/nodes/{node_id}` - Get node
- `PUT /api/graph/nodes/{node_id}` - Update node
- `DELETE /api/graph/nodes/{node_id}` - Delete node
- `POST /api/graph/edges` - Create edge
- `POST /api/graph/nodes/search` - Search nodes
- `GET /api/graph/stats` - Graph statistics

### Analysis & Intelligence
- `POST /api/analysis/attack_paths` - Find attack paths
- `POST /api/analysis/attack_surface` - Analyze attack surface
- `GET /api/analysis/attack_vectors/{node_id}` - Get attack vectors
- `POST /api/analysis/vulnerability_mapping` - Map vulnerabilities
- `POST /api/analysis/threat_attribution` - Analyze threat attribution
- `POST /api/analysis/communities` - Detect communities
- `POST /api/analysis/centrality` - Calculate centrality
- `GET /api/analysis/metrics` - Graph metrics
- `POST /api/analysis/anomalies` - Detect anomalies

### Advanced Analytics
- `POST /api/analysis/patterns` - Analyze security patterns
- `POST /api/analysis/intelligence_correlation` - Correlate intelligence
- `POST /api/analysis/trends` - Analyze trends
- `POST /api/analysis/comprehensive_risk` - Calculate risk scores
- `GET /api/analysis/analytics_report` - Export analytics report

### Maintenance
- `POST /api/analysis/backup` - Create backup
- `POST /api/analysis/restore` - Restore backup
- `GET /api/analysis/health` - Health check
- `POST /api/analysis/optimize` - Optimize graph
- `GET /api/analysis/maintenance_status` - Maintenance status

### Visualization
- `POST /api/visualization/layout` - Generate layout
- `POST /api/visualization/subgraph` - Generate subgraph
- `POST /api/visualization/path` - Visualize path
- `POST /api/visualization/network_map` - Generate network map
- `POST /api/visualization/export` - Export graph
- `GET /api/visualization/styling/node_types` - Node styling
- `GET /api/visualization/styling/edge_types` - Edge styling

## Graph Schema

### Node Types
- **HOST** - Computer systems, servers, workstations
- **SERVICE** - Network services and applications
- **VULNERABILITY** - Security vulnerabilities (CVEs)
- **USER** - User accounts and identities
- **CREDENTIAL** - Authentication credentials
- **ATTACK** - Attack instances and techniques
- **THREAT** - Threat actors and campaigns
- **NETWORK** - Network segments and subnets
- **DOMAIN** - DNS domains and subdomains
- **ARTIFACT** - Files, IOCs, and evidence

### Relationship Types
- **AFFECTS** - Vulnerability affects asset
- **EXPLOITS** - Attack exploits vulnerability
- **TARGETS** - Threat targets asset
- **CONNECTS_TO** - Network connectivity
- **USES** - Entity uses credential/tool
- **HOSTS** - Host runs service
- **BELONGS_TO** - Entity belongs to group/network
- **COMMUNICATES** - Communication relationship
- **CORRELATES** - Intelligence correlation

## Performance Considerations

### Scalability
- **Neo4j Clustering**: Support for Neo4j causal clusters
- **Read Replicas**: Read-only replicas for query scaling
- **Connection Pooling**: Efficient connection management
- **Query Optimization**: Indexed queries and query planning

### Caching Strategy
- **Node Caching**: Frequently accessed nodes cached in Redis
- **Query Result Caching**: Complex query results cached
- **Pattern Caching**: Security pattern detection results cached
- **Analytics Caching**: Analytics computations cached with TTL

### Memory Management
- **Lazy Loading**: Large graphs loaded incrementally
- **Pagination**: API responses paginated for large result sets
- **Batch Processing**: Bulk operations for performance
- **Garbage Collection**: Automatic cleanup of stale data

## Security Features

### Data Protection
- **Encryption at Rest**: Neo4j enterprise encryption
- **Transport Security**: TLS for all connections
- **Access Controls**: Role-based access control
- **Audit Logging**: Comprehensive audit trails

### Schema Validation
- **Input Validation**: All graph data validated against schema
- **Type Safety**: Strong typing for nodes and relationships
- **Constraint Enforcement**: Database constraints and indexes
- **Data Integrity**: Referential integrity checks

## Monitoring & Observability

### Health Monitoring
- **Connectivity Checks**: Database connection health
- **Performance Monitoring**: Query performance metrics
- **Data Integrity**: Schema validation and orphan detection
- **Storage Usage**: Disk usage and growth monitoring

### Metrics & Alerting
- **Graph Metrics**: Node/edge counts, density, connectivity
- **Performance Metrics**: Query latency, throughput
- **Error Rates**: Failed operations and exceptions
- **Custom Alerts**: Configurable alerting thresholds

## Best Practices

### Development
1. **Schema First**: Define schema before creating data
2. **Validation**: Always validate input data
3. **Error Handling**: Comprehensive error handling
4. **Testing**: Unit and integration tests
5. **Documentation**: Document all custom node/edge types

### Operations
1. **Regular Backups**: Automated daily backups
2. **Health Checks**: Regular health monitoring
3. **Performance Tuning**: Monitor and optimize queries
4. **Capacity Planning**: Monitor growth and plan capacity
5. **Security Updates**: Keep dependencies updated

### Data Management
1. **Cleanup**: Regular cleanup of orphaned nodes
2. **Archival**: Archive old data based on retention policies
3. **Indexing**: Maintain proper indexes for performance
4. **Relationships**: Minimize unnecessary relationships
5. **Normalization**: Avoid data duplication

## Troubleshooting

### Common Issues

#### Neo4j Connection Issues
```bash
# Check Neo4j service
systemctl status neo4j

# Check connectivity
curl -u neo4j:password http://localhost:7474/db/data/

# View logs
tail -f /var/log/neo4j/neo4j.log
```

#### Performance Issues
```python
# Enable query logging
graph_manager.config.neo4j_debug = True

# Check slow queries
await graph_manager.get_query_metrics()

# Optimize with indexes
CREATE INDEX FOR (n:Host) ON (n.ip)
```

#### Memory Issues
```python
# Check graph size
stats = await graph_manager.get_graph_stats()
print(f"Nodes: {stats['node_count']}, Edges: {stats['edge_count']}")

# Clear caches
await graph_manager.clear_cache()
```

### Support & Contributing

For issues, feature requests, or contributions:
1. Check existing issues in the repository
2. Create detailed bug reports with reproduction steps
3. Follow the coding standards and add tests
4. Update documentation for new features

## License

This knowledge graph implementation is part of the Aetherveil Sentinel project.