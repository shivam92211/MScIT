pragma solidity ^0.8.0;

contract Test {
    // Demonstrates modular addition: (7 + 3) % 3 = 10 % 3 = 1
    function CallAddMod() public pure returns (uint) {
        return addmod(7, 3, 3);
    }

    // Demonstrates modular multiplication: (7 * 3) % 3 = 21 % 3 = 0
    function CallMulMod() public pure returns (uint) {
        return mulmod(7, 3, 3);
    }
}
