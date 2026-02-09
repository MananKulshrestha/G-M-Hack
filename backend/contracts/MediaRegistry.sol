// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title MediaRegistry
 * @dev A simple digital notary to store verification results on-chain.
 */
contract MediaRegistry {
    
    // Define the structure of a record
    struct MediaRecord {
        string status;      // e.g., "REAL", "FAKE", "SUSPICIOUS"
        uint256 timestamp;  // Block timestamp
        address validator;  // The wallet address of the AI system
    }

    // Mapping from File Hash (ID) -> Record
    mapping(string => MediaRecord) public records;

    // Event emitted when a new record is added (useful for frontend listeners)
    event Verified(string indexed fileHash, string status, uint256 timestamp);

    /**
     * @dev Stores a verification result on the blockchain.
     * @param _fileHash The SHA-256 hash of the file (acts as the ID).
     * @param _status The result from the AI model.
     */
    function registerMedia(string memory _fileHash, string memory _status) public {
        // Update the record
        records[_fileHash] = MediaRecord({
            status: _status,
            timestamp: block.timestamp,
            validator: msg.sender
        });

        // Broadcast the event
        emit Verified(_fileHash, _status, block.timestamp);
    }

    /**
     * @dev Helper to retrieve data (optional, as mapping is public).
     */
    function getMediaStatus(string memory _fileHash) public view returns (string memory, uint256, address) {
        MediaRecord memory record = records[_fileHash];
        return (record.status, record.timestamp, record.validator);
    }
}