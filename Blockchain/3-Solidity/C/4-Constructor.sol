// SPDX-License-Identifier: MIT
pragma solidity ^0.5.0;

// Creating a contract
contract ConstructorExample {
    string str;

    // Constructor runs once at deployment
    constructor() public {
        str = "Shankar Narayan College";
    }

    // Getter function to retrieve the stored string
    function getValue() public view returns (string memory) {
        return str;
    }
}
