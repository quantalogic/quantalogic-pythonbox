# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Enhanced .gitignore with comprehensive Python development exclusions
- Added coverage for cache files, build artifacts, and temporary files

### Changed

- Improved project file organization and cleanup procedures
- Enhanced development environment configuration

## [0.9.22] - 2025-06-28

### Added

- Enhanced async generator functionality with proper stateful implementation
- Improved StatefulAsyncGenerator with better asend/athrow support
- Added comprehensive exception handling for async generators
- Generator context management with proper stack tracking
- Support for nested generator execution

### Fixed

- Fixed version inconsistencies in pyproject.toml
- Improved async generator parameter binding and validation
- Better handling of yield expressions in different statement contexts
- Enhanced error propagation in async generator athrow method

### Changed

- Refactored AsyncGeneratorFunction for better maintainability
- Improved generator resumption logic after asend/athrow operations
- Enhanced generator lifecycle management

## [0.9.12] - Previous Release

### Previous changes

- Various improvements and bug fixes (see git history for details)
