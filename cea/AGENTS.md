# Configuration Parameters

## Main API

- `Parameter.decode(value: str) → Any` - Parse value from config file (lenient)
- `Parameter.encode(value: Any) → str` - Validate value before saving (strict)
- `ChoiceParameter._choices → list[str]` - Available options (can be dynamic via `@property`)

## Key Pattern: decode() vs encode()

### ✅ DO: Separate parsing from validation

```python
class MyParameter(Parameter):
    def decode(self, value):
        """Parse - security checks only"""
        if not value:
            return ""
        value = value.strip()

        # Only validate security concerns (path traversal, injection, etc.)
        if self._has_security_issue(value):
            raise ValueError("Security violation")

        return value  # Don't check business rules

    def encode(self, value):
        """Validate - all business rules"""
        if not value or not value.strip():
            raise ValueError("Value required")

        value = value.strip()

        # Security check
        if self._has_security_issue(value):
            raise ValueError("Security violation")

        # Business rule check
        if self._resource_exists(value):
            raise ValueError(f"Resource '{value}' already exists")

        return value
```

### ❌ DON'T: Validate business rules in decode()

```python
def decode(self, value):
    # ❌ Expensive I/O on every config load
    if not self._resource_exists(value):
        raise ValueError("Resource not found")

    # ❌ Breaks loading old configs when resources deleted
    if self._check_collision(value):
        raise ValueError("Already exists")

    return value
```

**Why**: decode() is called when loading config files - must be lenient and fast.

## Dynamic Choices

### ✅ DO: Use @property for dynamic choices

```python
class DynamicChoiceParameter(ChoiceParameter):
    def initialize(self, parser):
        self.depends_on = ['other-param']  # Declare dependencies

    @property
    def _choices(self):
        """Scan resources on each access"""
        return self._get_available_options()

    def _get_available_options(self):
        # Scan filesystem/database for available options
        if not self._can_scan():
            return []

        # Return list of valid choices
        return self._scan_resources()
```

## Validation Helpers

Extract shared validation into helpers:

```python
def _validate_security(self, value):
    """Security checks (used by encode AND decode)"""
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    if any(char in value for char in invalid_chars):
        raise ValueError("Invalid characters")

def _validate_business(self, value):
    """Business rules (used by encode ONLY)"""
    if self._resource_exists(value):
        raise ValueError("Already exists")

def decode(self, value):
    return self._validate_security(value.strip())

def encode(self, value):
    self._validate_security(value.strip())
    self._validate_business(value)
    return value
```

## Common Pitfalls

1. **Validating in decode()** → Fragile config loading
2. **No dependency declaration** → Dynamic choices don't update
3. **Caching without invalidation** → Stale options
4. **Mixing security/business validation** → Security checks in both, business in encode only

## Related Files

- `config.py` - All parameter classes (PathParameter, ChoiceParameter, etc.)
- `config.pyi` - Type stubs (regenerate: `pixi run python cea/utilities/config_type_generator.py`)
- `default.config` - Default values for all parameters
- `interfaces/dashboard/api/tools.py` - Validation API endpoints (`validate_field`, `get_parameter_metadata`)

## Script + Config Pattern

- Register new CLI scripts in `scripts.yml` with `interfaces: [cli]` and a module exposing `main(config: Configuration)`.
- Add a dedicated section in `default.config` for script options (example: `[ucea]`), using unique parameter names to avoid collisions with other sections.
- Keep default behaviour explicit in `default.config`; for UCEA, `workflow1-only = true` means the default run executes workflow 1 only unless the user sets it to `false`.
- Regenerate `config.pyi` after config schema changes.

## Height Enrichment Parameters

- `zone-helper:building-height-gpkg` and `zone-helper:ine-fallback-max-distance-m` control optional INE point-height enrichment for `zone-helper`.
- `surroundings-helper:building-height-gpkg` and `surroundings-helper:ine-fallback-max-distance-m` control the same enrichment for `surroundings-helper`.
- Current fallback policy is: direct inside-point match first, then 2-nearest average when second-nearest distance is within threshold, then radius-median fallback from INE points within 50 m of each unmatched building.
- Default `ine-fallback-max-distance-m` is `20` for both zone and surroundings helpers.
- Keep these parameters nullable so behaviour remains unchanged when no GeoPackage is configured.
