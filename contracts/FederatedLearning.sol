// SPDX-License-Identifier: MIT
pragma solidity >=0.4.22 <0.9.0;

contract FederatedLearning {
    
    struct Device {
        address deviceAddress;
        string deviceId;
        uint256 reputation;
        bool isRegistered;
        uint256 totalContributions;
    }
    
    struct ModelUpdate {
        address contributor;
        string modelHash;
        uint256 round;
        uint256 timestamp;
        bool validated;
        uint256 accuracy;
        bytes signature;
        bool signatureVerified;
    }
    
    address public owner;
    uint256 public currentRound;
    uint256 public deviceCount;
    
    mapping(address => Device) public devices;
    mapping(uint256 => ModelUpdate[]) public roundUpdates;
    mapping(uint256 => string) public globalModelHashes;
    
    address[] public registeredDevices;
    
    event DeviceRegistered(address indexed device, string deviceId);
    event ModelUpdateSubmitted(address indexed device, uint256 round, string modelHash);
    event ModelValidated(address indexed device, uint256 round, bool passed);
    event GlobalModelUpdated(uint256 round, string modelHash);
    event ReputationUpdated(address indexed device, uint256 newReputation);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this");
        _;
    }
    
    modifier onlyRegistered() {
        require(devices[msg.sender].isRegistered, "Device not registered");
        _;
    }
    
    constructor() {
        owner = msg.sender;
        currentRound = 0;
        deviceCount = 0;
    }
    
    function registerDevice(string memory _deviceId) public {
        require(!devices[msg.sender].isRegistered, "Device already registered");
        
        devices[msg.sender] = Device({
            deviceAddress: msg.sender,
            deviceId: _deviceId,
            reputation: 100,
            isRegistered: true,
            totalContributions: 0
        });
        
        registeredDevices.push(msg.sender);
        deviceCount++;
        
        emit DeviceRegistered(msg.sender, _deviceId);
    }
    
    function submitModelUpdate(string memory _modelHash, uint256 _accuracy) public onlyRegistered {
        ModelUpdate memory update = ModelUpdate({
            contributor: msg.sender,
            modelHash: _modelHash,
            round: currentRound,
            timestamp: block.timestamp,
            validated: false,
            accuracy: _accuracy,
            signature: "",
            signatureVerified: false
        });

        roundUpdates[currentRound].push(update);
        devices[msg.sender].totalContributions++;

        emit ModelUpdateSubmitted(msg.sender, currentRound, _modelHash);
    }

    function submitSignedModelUpdate(
        string memory _modelHash,
        uint256 _accuracy,
        bytes memory _signature
    ) public onlyRegistered {
        // Verify the signature matches msg.sender
        bytes32 messageHash = keccak256(abi.encodePacked(_modelHash));
        bytes32 ethSignedHash = keccak256(
            abi.encodePacked("\x19Ethereum Signed Message:\n32", messageHash)
        );

        address recovered = recoverSigner(ethSignedHash, _signature);
        require(recovered == msg.sender, "Invalid signature: signer does not match sender");

        ModelUpdate memory update = ModelUpdate({
            contributor: msg.sender,
            modelHash: _modelHash,
            round: currentRound,
            timestamp: block.timestamp,
            validated: false,
            accuracy: _accuracy,
            signature: _signature,
            signatureVerified: true
        });

        roundUpdates[currentRound].push(update);
        devices[msg.sender].totalContributions++;

        emit ModelUpdateSubmitted(msg.sender, currentRound, _modelHash);
    }

    function recoverSigner(bytes32 _ethSignedHash, bytes memory _signature)
        internal pure returns (address)
    {
        require(_signature.length == 65, "Invalid signature length");
        bytes32 r;
        bytes32 s;
        uint8 v;
        assembly {
            r := mload(add(_signature, 32))
            s := mload(add(_signature, 64))
            v := byte(0, mload(add(_signature, 96)))
        }
        if (v < 27) {
            v += 27;
        }
        return ecrecover(_ethSignedHash, v, r, s);
    }
    
    function validateUpdate(address _device, uint256 _round, bool _passed) public onlyOwner {
        ModelUpdate[] storage updates = roundUpdates[_round];
        
        for (uint256 i = 0; i < updates.length; i++) {
            if (updates[i].contributor == _device) {
                updates[i].validated = _passed;
                
                if (_passed) {
                    if (devices[_device].reputation < 200) {
                        devices[_device].reputation += 10;
                    }
                } else {
                    if (devices[_device].reputation > 10) {
                        devices[_device].reputation -= 20;
                    }
                }
                
                emit ModelValidated(_device, _round, _passed);
                emit ReputationUpdated(_device, devices[_device].reputation);
                break;
            }
        }
    }
    
    function updateGlobalModel(string memory _modelHash) public onlyOwner {
        globalModelHashes[currentRound] = _modelHash;
        emit GlobalModelUpdated(currentRound, _modelHash);
        currentRound++;
    }
    
    function getDeviceReputation(address _device) public view returns (uint256) {
        return devices[_device].reputation;
    }
    
    function getRoundUpdateCount(uint256 _round) public view returns (uint256) {
        return roundUpdates[_round].length;
    }
    
    function getRegisteredDevices() public view returns (address[] memory) {
        return registeredDevices;
    }
    
    function isDeviceRegistered(address _device) public view returns (bool) {
        return devices[_device].isRegistered;
    }
}
